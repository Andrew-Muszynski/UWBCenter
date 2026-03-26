// ============================================================
// UWB Anchor (A1) — SS-TWR Responder + BLE Telemetry + Remote Commands
// Board: Arduino Nano 33 BLE Sense Lite + DWM1000 shield
//
// Listens for POLL messages, sends two responses (immediate +
// reply-delay payload), and broadcasts its own RX diagnostics
// over BLE so the Pi dashboard can monitor anchor health too.
//
// NEW: Accepts commands over BLE to change antenna delay and
//      other settings at runtime from the dashboard.
//
// BLE device name format: A1, A2, A3 ...  (set DEVICE_ID below)
// ============================================================

#include <SPI.h>
#include <DW1000.h>
#include <ArduinoBLE.h>

// ---------- identity ----------
#define DEVICE_ID   1                     // 1 -> "A1", 2 -> "A2", etc.
#define DEVICE_NAME "A1"                  // must match DEVICE_ID

// ---------- DW1000 wiring ----------
const uint8_t PIN_CS  = 10;
const uint8_t PIN_IRQ = 2;
const uint8_t PIN_RST = 3;

// ---------- TWR message types ----------
#define MSG_POLL     0x01
#define MSG_RESPONSE 0x02

// ---------- watchdog ----------
#define WATCHDOG_MS  5000UL

// ---------- runtime settings ----------
uint16_t antennaDelay = 0;                // current antenna delay register value

// ---------- BLE UUIDs ----------
#define UWB_SERVICE_UUID   "19b10010-e8f2-537e-4f6c-d104768a1214"
#define ANCHOR_CHAR_UUID   "19b10012-e8f2-537e-4f6c-d104768a1214"
#define CMD_CHAR_UUID      "19b10013-e8f2-537e-4f6c-d104768a1214"

// ---------- BLE data frame ----------
// 33 bytes total
struct __attribute__((packed)) AnchorFrame {
  uint8_t  tag_id;
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
  int32_t  reply_delay_lo;
  uint8_t  reply_delay_hi;
  uint8_t  flags;
};

struct RawDiag {
  float    rx_power;
  float    fp_power;
  float    quality;
  uint16_t std_noise;
  uint16_t fp_ampl1;
  uint16_t fp_ampl2;
  uint16_t fp_ampl3;
  uint16_t cir_power;
  uint16_t rxpacc;
};

// ---------- globals ----------
byte rxBuffer[20];
byte txBuffer[20];
uint16_t rangeSeq  = 0;
uint32_t lastGoodMs = 0;

BLEService        uwbService(UWB_SERVICE_UUID);
BLECharacteristic anchorChar(ANCHOR_CHAR_UUID, BLERead | BLENotify, sizeof(AnchorFrame));
BLECharacteristic cmdChar(CMD_CHAR_UUID, BLEWrite | BLERead, 32);

// ---------- helpers ----------

static inline void clearStatusAll() {
  byte clear[5] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
  DW1000.writeBytes(SYS_STATUS, NO_SUB, clear, 5);
}

void startReceiver() {
  DW1000.newReceive();
  DW1000.setDefaults();
  DW1000.receivePermanently(true);
  DW1000.startReceive();
}

void dwmSoftReset() {
  Serial.println(F("[RST] DWM soft-reset..."));
  pinMode(PIN_RST, OUTPUT);
  digitalWrite(PIN_RST, LOW);
  delay(2);
  pinMode(PIN_RST, INPUT);
  delay(10);

  DW1000.begin(PIN_IRQ, PIN_RST);
  DW1000.select(PIN_CS);

  DW1000.newConfiguration();
  DW1000.setDefaults();
  DW1000.setDeviceAddress(DEVICE_ID);
  DW1000.setNetworkId(10);
  DW1000.enableMode(DW1000.MODE_LONGDATA_RANGE_LOWPOWER);
  DW1000.commitConfiguration();

  // Re-apply antenna delay if set
  if (antennaDelay > 0) {
    byte buf[2] = { (byte)(antennaDelay & 0xFF), (byte)((antennaDelay >> 8) & 0xFF) };
    DW1000.writeBytes(0x18, 0x00, buf, 2);
    DW1000.writeBytes(0x2E, 0x1804, buf, 2);
  }

  byte pmsc[4];
  DW1000.readBytes(0x36, 0x04, pmsc, 4);
  pmsc[1] &= ~0x18;
  DW1000.writeBytes(0x36, 0x04, pmsc, 4);

  byte zeros[4] = {0, 0, 0, 0};
  DW1000.writeBytes(SYS_MASK, NO_SUB, zeros, 4);

  startReceiver();
  Serial.println(F("[RST] Done. Listening..."));
}

RawDiag readRxDiagnostics() {
  RawDiag d;
  d.rx_power = DW1000.getReceivePower();
  d.fp_power = DW1000.getFirstPathPower();
  d.quality  = DW1000.getReceiveQuality();

  byte fqual[8];
  DW1000.readBytes(0x12, NO_SUB, fqual, 8);
  d.std_noise = (uint16_t)fqual[0] | ((uint16_t)fqual[1] << 8);
  d.fp_ampl2  = (uint16_t)fqual[2] | ((uint16_t)fqual[3] << 8);
  d.fp_ampl3  = (uint16_t)fqual[4] | ((uint16_t)fqual[5] << 8);
  d.cir_power = (uint16_t)fqual[6] | ((uint16_t)fqual[7] << 8);

  byte rxtime[9];
  DW1000.readBytes(0x15, NO_SUB, rxtime, 9);
  d.fp_ampl1 = (uint16_t)rxtime[7] | ((uint16_t)rxtime[8] << 8);

  byte rxfinfo[4];
  DW1000.readBytes(0x10, NO_SUB, rxfinfo, 4);
  uint32_t fi = (uint32_t)rxfinfo[0]        |
                ((uint32_t)rxfinfo[1] <<  8) |
                ((uint32_t)rxfinfo[2] << 16) |
                ((uint32_t)rxfinfo[3] << 24);
  d.rxpacc = (uint16_t)((fi >> 20) & 0xFFF);

  return d;
}

// ---------- Apply antenna delay to DW1000 ----------
void setAntennaDelay(uint16_t value) {
  antennaDelay = value;
  byte buf[2] = { (byte)(value & 0xFF), (byte)((value >> 8) & 0xFF) };
  DW1000.writeBytes(0x18, 0x00, buf, 2);
  DW1000.writeBytes(0x2E, 0x1804, buf, 2);
  Serial.print(F("[CMD] Antenna delay set to: "));
  Serial.println(value);
}

// ---------- Process BLE command ----------
void processCommand(const char* cmd) {
  Serial.print(F("[CMD] Received: "));
  Serial.println(cmd);

  // AD:<value> = set antenna delay
  if (strncmp(cmd, "AD:", 3) == 0) {
    uint16_t val = (uint16_t)atoi(cmd + 3);
    setAntennaDelay(val);
    char resp[32];
    snprintf(resp, sizeof(resp), "AD:%u OK", val);
    cmdChar.writeValue((uint8_t*)resp, strlen(resp));
  }
  // ST = status query
  else if (strncmp(cmd, "ST", 2) == 0) {
    char resp[32];
    snprintf(resp, sizeof(resp), "AD:%u", antennaDelay);
    cmdChar.writeValue((uint8_t*)resp, strlen(resp));
    Serial.print(F("[CMD] Status: "));
    Serial.println(resp);
  }
}

// ============================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println(F("=== UWB Anchor + BLE + CMD  [" DEVICE_NAME "] ==="));

  DW1000.begin(PIN_IRQ, PIN_RST);
  DW1000.select(PIN_CS);

  DW1000.newConfiguration();
  DW1000.setDefaults();
  DW1000.setDeviceAddress(DEVICE_ID);
  DW1000.setNetworkId(10);
  DW1000.enableMode(DW1000.MODE_LONGDATA_RANGE_LOWPOWER);
  DW1000.commitConfiguration();

  byte pmsc[4];
  DW1000.readBytes(0x36, 0x04, pmsc, 4);
  pmsc[1] &= ~0x18;
  DW1000.writeBytes(0x36, 0x04, pmsc, 4);

  byte zeros[4] = {0, 0, 0, 0};
  DW1000.writeBytes(SYS_MASK, NO_SUB, zeros, 4);

  if (!BLE.begin()) {
    Serial.println(F("BLE init failed!"));
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
  Serial.println(F("BLE advertising. Listening for POLLs...\n"));
  startReceiver();
  lastGoodMs = millis();
}

// ============================================================
void loop() {
  BLE.poll();

  // Check for incoming BLE commands
  if (cmdChar.written()) {
    char buf[33] = {0};
    int len = cmdChar.valueLength();
    if (len > 32) len = 32;
    memcpy(buf, cmdChar.value(), len);
    buf[len] = '\0';
    processCommand(buf);
  }

  // ---- Background watchdog ----
  if ((uint32_t)(millis() - lastGoodMs) > WATCHDOG_MS) {
    Serial.print(F("[WDT] No exchange for "));
    Serial.print(WATCHDOG_MS);
    Serial.println(F(" ms -- resetting DWM"));
    dwmSoftReset();
    lastGoodMs = millis();
    return;
  }

  byte status[5];
  DW1000.readBytes(SYS_STATUS, NO_SUB, status, 5);

  bool dataReady = (status[1] & 0x20);
  bool goodCRC   = (status[1] & 0x40);

  if (dataReady && goodCRC) {
    DW1000Time t2;
    DW1000.getReceiveTimestamp(t2);

    RawDiag diag = readRxDiagnostics();

    uint16_t len = DW1000.getDataLength();
    if (len > sizeof(rxBuffer)) len = sizeof(rxBuffer);
    DW1000.getData(rxBuffer, len);

    clearStatusAll();

    if (rxBuffer[0] == MSG_POLL) {
      rangeSeq++;

      uint8_t tagId = 0;
      if (len >= 2) tagId = rxBuffer[1];

      // ---- Send RESPONSE #1 ----
      txBuffer[0] = MSG_RESPONSE;
      txBuffer[1] = DEVICE_ID;

      DW1000.newTransmit();
      DW1000.setDefaults();
      DW1000.setData(txBuffer, 2);
      DW1000.startTransmit();

      unsigned long start = millis();
      bool tx1ok = false;
      while ((millis() - start) < 50) {
        DW1000.readBytes(SYS_STATUS, NO_SUB, status, 5);
        if (status[0] & 0x80) { tx1ok = true; break; }
        delayMicroseconds(50);
      }
      if (!tx1ok) {
        Serial.println(F("[ERR] TX1 timeout -- resetting DWM"));
        clearStatusAll();
        dwmSoftReset();
        lastGoodMs = millis();
        return;
      }

      DW1000Time t3;
      DW1000.getTransmitTimestamp(t3);

      int64_t replyDelay = t3.getTimestamp() - t2.getTimestamp();

      clearStatusAll();

      // ---- Send RESPONSE #2 (reply delay) ----
      txBuffer[0] = MSG_RESPONSE;
      txBuffer[1] = (replyDelay >>  0) & 0xFF;
      txBuffer[2] = (replyDelay >>  8) & 0xFF;
      txBuffer[3] = (replyDelay >> 16) & 0xFF;
      txBuffer[4] = (replyDelay >> 24) & 0xFF;
      txBuffer[5] = (replyDelay >> 32) & 0xFF;

      DW1000.newTransmit();
      DW1000.setDefaults();
      DW1000.setData(txBuffer, 6);
      DW1000.startTransmit();

      start = millis();
      bool tx2ok = false;
      while ((millis() - start) < 50) {
        DW1000.readBytes(SYS_STATUS, NO_SUB, status, 5);
        if (status[0] & 0x80) { tx2ok = true; break; }
        delayMicroseconds(50);
      }
      if (!tx2ok) {
        Serial.println(F("[ERR] TX2 timeout -- resetting DWM"));
        clearStatusAll();
        dwmSoftReset();
        lastGoodMs = millis();
        return;
      }
      clearStatusAll();

      // ---- Build & send BLE frame ----
      AnchorFrame frame;
      frame.tag_id         = tagId;
      frame.seq            = rangeSeq;
      frame.rx_power       = diag.rx_power;
      frame.fp_power       = diag.fp_power;
      frame.quality        = diag.quality;
      frame.std_noise      = diag.std_noise;
      frame.fp_ampl1       = diag.fp_ampl1;
      frame.fp_ampl2       = diag.fp_ampl2;
      frame.fp_ampl3       = diag.fp_ampl3;
      frame.cir_power      = diag.cir_power;
      frame.rxpacc         = diag.rxpacc;
      frame.reply_delay_lo = (int32_t)(replyDelay & 0xFFFFFFFFLL);
      frame.reply_delay_hi = (uint8_t)((replyDelay >> 32) & 0xFF);
      frame.flags          = 0x01;

      anchorChar.writeValue((uint8_t*)&frame, sizeof(AnchorFrame));

      lastGoodMs = millis();

      Serial.print(F("A")); Serial.print(DEVICE_ID);
      Serial.print(F("<-T")); Serial.print(tagId);
      Serial.print(F(" #")); Serial.print(rangeSeq);
      Serial.print(F("  RX="));   Serial.print(diag.rx_power, 1);
      Serial.print(F("  FP="));   Serial.print(diag.fp_power, 1);
      Serial.print(F("  Q="));    Serial.print(diag.quality, 1);
      Serial.print(F("  SN="));   Serial.print(diag.std_noise);
      Serial.print(F("  A1="));   Serial.print(diag.fp_ampl1);
      Serial.print(F("  A2="));   Serial.print(diag.fp_ampl2);
      Serial.print(F("  A3="));   Serial.print(diag.fp_ampl3);
      Serial.print(F("  PACC=")); Serial.print(diag.rxpacc);
      Serial.print(F("  RD="));   Serial.print((long)replyDelay);
      Serial.println();
    }

    startReceiver();

  } else if (dataReady) {
    clearStatusAll();
    startReceiver();
  }

  delayMicroseconds(100);
}
