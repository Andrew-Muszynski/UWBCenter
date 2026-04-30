// ============================================================
// UWB Anchor (A1) — SS-TWR Responder + BLE Telemetry
// Board: Arduino Nano 33 BLE Sense Lite + DWM1000 shield
//
// Built directly on the working SS_RANGING_ANCHOR_33 protocol:
//   1. Tag  sends POLL  [0x01, tag_id]
//   2. Anchor sends RESP1 [0x02, anchor_id]   <- immediate, t3 captured here
//   3. Anchor sends RESP2 [0x02, rd0..rd4]    <- reply delay (T3-T2)
//
// BLE frame (33 bytes) matches Python "<BHfffHHHHHHiBB"
// Device name "A1" is discovered by uwb_dashboard.py
// ============================================================

#include <SPI.h>
#include <DW1000.h>
#include <ArduinoBLE.h>

#define DEVICE_ID   1
#define DEVICE_NAME "A1"

const uint8_t PIN_CS  = 10;
const uint8_t PIN_IRQ = 2;
const uint8_t PIN_RST = 3;

#define MSG_POLL     0x01
#define MSG_RESPONSE 0x02
#define WATCHDOG_MS  8000UL

// BLE UUIDs — must match uwb_dashboard.py exactly
#define UWB_SERVICE_UUID  "19b10010-e8f2-537e-4f6c-d104768a1214"
#define ANCHOR_CHAR_UUID  "19b10012-e8f2-537e-4f6c-d104768a1214"
#define CMD_CHAR_UUID     "19b10013-e8f2-537e-4f6c-d104768a1214"

// BLE data frame — layout matches Python: "<BHfffHHHHHHiBB" = 33 bytes
struct __attribute__((packed)) AnchorFrame {
  uint8_t  tag_id;          // which tag sent the POLL
  uint16_t seq;
  float    rx_power;
  float    fp_power;
  float    quality;
  uint16_t std_noise;
  uint16_t fp_ampl1;
  uint16_t fp_ampl2;
  uint16_t fp_ampl3;
  uint16_t cir_power;
  uint16_t rxpacc;
  int32_t  reply_delay_lo;  // actual T3-T2 lower 32 bits
  uint8_t  reply_delay_hi;  // upper 8 bits
  uint8_t  flags;
};

byte     rxBuffer[20];
byte     txBuffer[20];
uint16_t rangeSeq   = 0;
uint32_t lastGoodMs = 0;
uint16_t antennaDelay = 16384;  // tune via BLE "AD:NNNN"

BLEService        uwbService(UWB_SERVICE_UUID);
BLECharacteristic anchorChar(ANCHOR_CHAR_UUID, BLERead | BLENotify, sizeof(AnchorFrame));
BLECharacteristic cmdChar(CMD_CHAR_UUID, BLEWrite | BLERead, 32);

// ---------- helpers ----------

void applyAntennaDelay() {
  byte buf[2] = { (byte)(antennaDelay & 0xFF), (byte)((antennaDelay >> 8) & 0xFF) };
  DW1000.writeBytes(0x18, 0x00, buf, 2);
  DW1000.writeBytes(0x2E, 0x1804, buf, 2);
}

void dwInit() {
  DW1000.begin(PIN_IRQ, PIN_RST);
  DW1000.select(PIN_CS);
  DW1000.newConfiguration();
  DW1000.setDefaults();
  DW1000.setDeviceAddress(DEVICE_ID);
  DW1000.setNetworkId(10);
  DW1000.enableMode(DW1000.MODE_LONGDATA_RANGE_LOWPOWER);
  DW1000.commitConfiguration();
  applyAntennaDelay();
  // disable auto-sleep
  byte pmsc[4];
  DW1000.readBytes(0x36, 0x04, pmsc, 4);
  pmsc[1] &= ~0x18;
  DW1000.writeBytes(0x36, 0x04, pmsc, 4);
  // disable interrupts (polled mode)
  byte zeros[4] = {0, 0, 0, 0};
  DW1000.writeBytes(SYS_MASK, NO_SUB, zeros, 4);
}

void startReceiver() {
  DW1000.idle();
  DW1000.newReceive();
  DW1000.receivePermanently(true);
  DW1000.startReceive();
}

static inline bool waitForTxDone(uint16_t timeoutMs) {
  byte st[5];
  unsigned long t = millis();
  while (millis() - t < timeoutMs) {
    DW1000.readBytes(SYS_STATUS, NO_SUB, st, 5);
    if (st[0] & 0x80) return true;
    delayMicroseconds(50);
  }
  return false;
}

void readDiagnostics(float &rxp, float &fpp, float &qual,
                     uint16_t &sn, uint16_t &a1, uint16_t &a2,
                     uint16_t &a3, uint16_t &cp, uint16_t &pacc) {
  rxp  = DW1000.getReceivePower();
  fpp  = DW1000.getFirstPathPower();
  qual = DW1000.getReceiveQuality();
  byte fq[8];
  DW1000.readBytes(0x12, NO_SUB, fq, 8);
  sn = (uint16_t)fq[0] | ((uint16_t)fq[1] << 8);
  a2 = (uint16_t)fq[2] | ((uint16_t)fq[3] << 8);
  a3 = (uint16_t)fq[4] | ((uint16_t)fq[5] << 8);
  cp = (uint16_t)fq[6] | ((uint16_t)fq[7] << 8);
  byte rt[9];
  DW1000.readBytes(0x15, NO_SUB, rt, 9);
  a1 = (uint16_t)rt[7] | ((uint16_t)rt[8] << 8);
  byte fi[4];
  DW1000.readBytes(0x10, NO_SUB, fi, 4);
  uint32_t fiv = (uint32_t)fi[0] | ((uint32_t)fi[1] << 8)
               | ((uint32_t)fi[2] << 16) | ((uint32_t)fi[3] << 24);
  pacc = (uint16_t)((fiv >> 20) & 0xFFF);
}

void processCommand(const char* cmd) {
  if (strncmp(cmd, "AD:", 3) == 0) {
    antennaDelay = (uint16_t)atoi(cmd + 3);
    applyAntennaDelay();
    char resp[32];
    snprintf(resp, sizeof(resp), "AD:%u OK", antennaDelay);
    cmdChar.writeValue((uint8_t*)resp, strlen(resp));
    Serial.print("[CMD] AD="); Serial.println(antennaDelay);
  } else if (strncmp(cmd, "ST", 2) == 0) {
    char resp[32];
    snprintf(resp, sizeof(resp), "AD:%u", antennaDelay);
    cmdChar.writeValue((uint8_t*)resp, strlen(resp));
  }
}

// ============================================================
void setup() {
  Serial.begin(115200);
  // wait up to 2s for serial monitor; proceed regardless
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 2000) { delay(10); }

  Serial.println("=== UWB Anchor v3 + BLE [" DEVICE_NAME "] ===");

  dwInit();

  // SPI sanity check
  byte devId[4];
  DW1000.readBytes(0x00, 0x00, devId, 4);
  Serial.print("DEVID: 0x");
  for (int i = 3; i >= 0; i--) {
    if (devId[i] < 0x10) Serial.print("0");
    Serial.print(devId[i], HEX);
  }
  Serial.println();

  if (!BLE.begin()) {
    Serial.println("BLE init failed!");
    while (1) delay(1000);
  }
  BLE.setDeviceName(DEVICE_NAME);
  BLE.setLocalName(DEVICE_NAME);
  BLE.setAdvertisedService(uwbService);
  uwbService.addCharacteristic(anchorChar);
  uwbService.addCharacteristic(cmdChar);
  BLE.addService(uwbService);

  AnchorFrame z = {};
  anchorChar.writeValue((uint8_t*)&z, sizeof(AnchorFrame));
  char initStatus[32];
  snprintf(initStatus, sizeof(initStatus), "AD:%u", antennaDelay);
  cmdChar.writeValue((uint8_t*)initStatus, strlen(initStatus));

  BLE.advertise();
  Serial.println("BLE advertising. Listening for POLLs...\n");
  startReceiver();
  lastGoodMs = millis();
}

// ============================================================
void loop() {
  BLE.poll();

  if (cmdChar.written()) {
    char buf[33] = {0};
    int len = cmdChar.valueLength();
    if (len > 32) len = 32;
    memcpy(buf, cmdChar.value(), len);
    processCommand(buf);
  }

  // Watchdog — same logic as working SS_RANGING but with soft reinit
  if ((uint32_t)(millis() - lastGoodMs) > WATCHDOG_MS) {
    Serial.println("[WDT] No exchange — reinitialising DWM...");
    dwInit();
    startReceiver();
    lastGoodMs = millis();
    return;
  }

  byte status[5];
  DW1000.readBytes(SYS_STATUS, NO_SUB, status, 5);

  bool dataReady = (status[1] & 0x20);  // RXDFR
  bool goodCRC   = (status[1] & 0x40);  // RXFCG
  bool rxErr     = (status[1] & 0x90) || (status[2] & 0x05) || (status[3] & 0x20);

  if (rxErr) {
    byte clear[5] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    DW1000.writeBytes(SYS_STATUS, NO_SUB, clear, 5);
    startReceiver();
    return;
  }

  if (dataReady && goodCRC) {
    DW1000Time t2;
    DW1000.getReceiveTimestamp(t2);

    uint16_t len = DW1000.getDataLength();
    if (len > sizeof(rxBuffer)) len = sizeof(rxBuffer);
    DW1000.getData(rxBuffer, len);

    // Read diagnostics before clearing RX status
    float rxp, fpp, qual;
    uint16_t sn, a1, a2, a3, cp, pacc;
    readDiagnostics(rxp, fpp, qual, sn, a1, a2, a3, cp, pacc);

    byte clear[5] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    DW1000.writeBytes(SYS_STATUS, NO_SUB, clear, 5);

    if (rxBuffer[0] == MSG_POLL) {
      rangeSeq++;
      uint8_t tagId = (len >= 2) ? rxBuffer[1] : 0;

      // ---- RESP1: immediate, carries anchor ID so tag knows who responded ----
      txBuffer[0] = MSG_RESPONSE;
      txBuffer[1] = DEVICE_ID;

      DW1000.idle();
      DW1000.newTransmit();
      DW1000.setData(txBuffer, 2);
      DW1000.startTransmit();

      if (!waitForTxDone(50)) {
        Serial.println("[ERR] RESP1 TX timeout");
        DW1000.writeBytes(SYS_STATUS, NO_SUB, clear, 5);
        startReceiver();
        return;
      }

      DW1000Time t3;
      DW1000.getTransmitTimestamp(t3);
      int64_t replyDelay = t3.getTimestamp() - t2.getTimestamp();

      DW1000.writeBytes(SYS_STATUS, NO_SUB, clear, 5);

      // ---- RESP2: carries reply delay so tag can compute ToF ----
      txBuffer[0] = MSG_RESPONSE;
      txBuffer[1] = (replyDelay >>  0) & 0xFF;
      txBuffer[2] = (replyDelay >>  8) & 0xFF;
      txBuffer[3] = (replyDelay >> 16) & 0xFF;
      txBuffer[4] = (replyDelay >> 24) & 0xFF;
      txBuffer[5] = (replyDelay >> 32) & 0xFF;

      DW1000.idle();
      DW1000.newTransmit();
      DW1000.setData(txBuffer, 6);
      DW1000.startTransmit();
      waitForTxDone(50);
      DW1000.writeBytes(SYS_STATUS, NO_SUB, clear, 5);

      // ---- BLE frame ----
      AnchorFrame frame;
      frame.tag_id         = tagId;
      frame.seq            = rangeSeq;
      frame.rx_power       = rxp;
      frame.fp_power       = fpp;
      frame.quality        = qual;
      frame.std_noise      = sn;
      frame.fp_ampl1       = a1;
      frame.fp_ampl2       = a2;
      frame.fp_ampl3       = a3;
      frame.cir_power      = cp;
      frame.rxpacc         = pacc;
      frame.reply_delay_lo = (int32_t)(replyDelay & 0xFFFFFFFFLL);
      frame.reply_delay_hi = (uint8_t)((replyDelay >> 32) & 0xFF);
      frame.flags          = 0x01;
      anchorChar.writeValue((uint8_t*)&frame, sizeof(AnchorFrame));

      lastGoodMs = millis();

      Serial.print("A"); Serial.print(DEVICE_ID);
      Serial.print(" <- T"); Serial.print(tagId);
      Serial.print("  #"); Serial.print(rangeSeq);
      Serial.print("  RX="); Serial.print(rxp, 1);
      Serial.print("  FP="); Serial.print(fpp, 1);
      Serial.print("  Q=");  Serial.print(qual, 1);
      Serial.print("  SN="); Serial.print(sn);
      Serial.print("  RD="); Serial.println((long)replyDelay);
    } else {
      Serial.print("[WARN] non-POLL frame: 0x");
      Serial.println(rxBuffer[0], HEX);
    }

    startReceiver();

  } else if (dataReady) {
    byte clear[5] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    DW1000.writeBytes(SYS_STATUS, NO_SUB, clear, 5);
    startReceiver();
  }

  delay(1);
}
