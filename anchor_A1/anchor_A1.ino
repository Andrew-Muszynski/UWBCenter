// ============================================================
// UWB Anchor (A1) — SS-TWR Responder + BLE Telemetry + Remote Commands
// Board: Arduino Nano 33 BLE Sense Lite + DWM1000 shield
//
// v4: Raw-register delayed TX — bypasses the DW1000 library's
//     newTransmit()/setDefaults() overhead to avoid HPDWARN.
//     Fixed reply delay = 10 ms (generous margin).
//
// Protocol:
//   1. Tag sends POLL [0x01, tag_id]
//   2. Anchor receives, schedules delayed TX at T2 + 10ms
//   3. Anchor sends RESPONSE [0x02, anchor_id, reply_delay(5 bytes)]
//   4. Tag computes ToF from round_trip and reply_delay
//
// BLE device name format: A1, A2, A3 ...  (set DEVICE_ID below)
// ============================================================

#include <SPI.h>
#include <DW1000.h>
#include <ArduinoBLE.h>

// ---------- Serial gating (no-op when no USB host) ----------
#define SP(x)    do { if (Serial) Serial.print(x); } while (0)
#define SPLN(x)  do { if (Serial) Serial.println(x); } while (0)
#define SP2(x,y) do { if (Serial) Serial.print((x), (y)); } while (0)

// ---------- identity ----------
#define DEVICE_ID   1
#define DEVICE_NAME "A1"

// ---------- DW1000 wiring ----------
const uint8_t PIN_CS  = 10;
const uint8_t PIN_IRQ = 2;
const uint8_t PIN_RST = 3;

// ---------- TWR message types ----------
#define MSG_POLL     0x01
#define MSG_RESPONSE 0x02

// ---------- watchdog ----------
#define WATCHDOG_MS  5000UL

// ---------- fixed reply delay ----------
// 10 ms in DW1000 ticks: 10e-3 / 15.65e-12 = 638,977,636
#define FIXED_REPLY_DLY_TICKS  638977636ULL

// ---------- runtime settings ----------
uint16_t antennaDelay = 0;

// ---------- BLE UUIDs ----------
#define UWB_SERVICE_UUID   "19b10010-e8f2-537e-4f6c-d104768a1214"
#define ANCHOR_CHAR_UUID   "19b10012-e8f2-537e-4f6c-d104768a1214"
#define CMD_CHAR_UUID      "19b10013-e8f2-537e-4f6c-d104768a1214"

// ---------- BLE data frame ----------
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
  SPLN(F("[RST] DWM soft-reset..."));
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
  SPLN(F("[RST] Done. Listening..."));
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

void setAntennaDelay(uint16_t value) {
  antennaDelay = value;
  byte buf[2] = { (byte)(value & 0xFF), (byte)((value >> 8) & 0xFF) };
  DW1000.writeBytes(0x18, 0x00, buf, 2);
  DW1000.writeBytes(0x2E, 0x1804, buf, 2);
  SP(F("[CMD] Antenna delay set to: "));
  SPLN(value);
}

void processCommand(const char* cmd) {
  SP(F("[CMD] Received: "));
  SPLN(cmd);

  if (strncmp(cmd, "AD:", 3) == 0) {
    uint16_t val = (uint16_t)atoi(cmd + 3);
    setAntennaDelay(val);
    char resp[32];
    snprintf(resp, sizeof(resp), "AD:%u OK", val);
    cmdChar.writeValue((uint8_t*)resp, strlen(resp));
  }
  else if (strncmp(cmd, "ST", 2) == 0) {
    char resp[32];
    snprintf(resp, sizeof(resp), "AD:%u", antennaDelay);
    cmdChar.writeValue((uint8_t*)resp, strlen(resp));
    SP(F("[CMD] Status: "));
    SPLN(resp);
  }
}

// ============================================================
// Raw delayed transmit — bypasses library overhead
// ============================================================
bool rawDelayedTransmit(byte* data, uint8_t dataLen, uint64_t txTime) {
  // 1. Write TX data to TX_BUFFER (register 0x09)
  DW1000.writeBytes(0x09, NO_SUB, data, dataLen);

  // 2. Write TX frame length to TX_FCTRL (register 0x08)
  //    Bytes [0:6] = frame length (including 2-byte CRC added by DW1000)
  //    We need to set the length field = dataLen + 2 (for auto-CRC)
  uint16_t frameLen = dataLen + 2;
  byte txfctrl[5];
  DW1000.readBytes(0x08, NO_SUB, txfctrl, 5);
  txfctrl[0] = (byte)(frameLen & 0xFF);
  txfctrl[1] = (txfctrl[1] & 0xFC) | (byte)((frameLen >> 8) & 0x03);
  DW1000.writeBytes(0x08, NO_SUB, txfctrl, 5);

  // 3. Write DX_TIME (register 0x0A, 5 bytes) — the target TX timestamp
  byte dxTime[5];
  dxTime[0] = (txTime >>  0) & 0xFF;
  dxTime[1] = (txTime >>  8) & 0xFF;
  dxTime[2] = (txTime >> 16) & 0xFF;
  dxTime[3] = (txTime >> 24) & 0xFF;
  dxTime[4] = (txTime >> 32) & 0xFF;
  DW1000.writeBytes(0x0A, NO_SUB, dxTime, 5);

  // 4. Start delayed transmit: SYS_CTRL bit 1 (TXSTRT) + bit 2 (TXDLYE)
  byte ctrl[1] = { 0x06 };  // TXSTRT | TXDLYE
  DW1000.writeBytes(0x0D, NO_SUB, ctrl, 1);

  // 5. Wait for completion
  byte status[5];
  unsigned long start = millis();
  while ((millis() - start) < 25) {
    DW1000.readBytes(SYS_STATUS, NO_SUB, status, 5);
    // HPDWARN = bit 27 = byte 3 bit 3
    if (status[3] & 0x08) return false;
    // TXFRS = bit 7 = byte 0 bit 7
    if (status[0] & 0x80) return true;
    delayMicroseconds(50);
  }
  return false;  // timeout
}

// ============================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  SPLN(F("=== UWB Anchor v4 (raw delayed TX) + BLE + CMD  [" DEVICE_NAME "] ==="));

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
    SPLN(F("BLE init failed!"));
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
  SPLN(F("BLE advertising. Listening for POLLs...\n"));
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
    buf[len] = '\0';
    processCommand(buf);
  }

  if ((uint32_t)(millis() - lastGoodMs) > WATCHDOG_MS) {
    SP(F("[WDT] No exchange for "));
    SP(WATCHDOG_MS);
    SPLN(F(" ms -- resetting DWM"));
    dwmSoftReset();
    lastGoodMs = millis();
    return;
  }

  byte status[5];
  DW1000.readBytes(SYS_STATUS, NO_SUB, status, 5);

  bool dataReady = (status[1] & 0x20);
  bool goodCRC   = (status[1] & 0x40);

  if (dataReady && goodCRC) {
    // ---- FAST PATH: timestamp + data only ----
    DW1000Time t2;
    DW1000.getReceiveTimestamp(t2);

    uint16_t len = DW1000.getDataLength();
    if (len > sizeof(rxBuffer)) len = sizeof(rxBuffer);
    DW1000.getData(rxBuffer, len);

    clearStatusAll();

    if (rxBuffer[0] == MSG_POLL) {
      rangeSeq++;

      uint8_t tagId = 0;
      if (len >= 2) tagId = rxBuffer[1];

      // ---- Build response packet ----
      int64_t replyDelay = (int64_t)FIXED_REPLY_DLY_TICKS;

      txBuffer[0] = MSG_RESPONSE;
      txBuffer[1] = DEVICE_ID;
      txBuffer[2] = (replyDelay >>  0) & 0xFF;
      txBuffer[3] = (replyDelay >>  8) & 0xFF;
      txBuffer[4] = (replyDelay >> 16) & 0xFF;
      txBuffer[5] = (replyDelay >> 24) & 0xFF;
      txBuffer[6] = (replyDelay >> 32) & 0xFF;

      // ---- Schedule delayed TX ----
      uint64_t txTime = (uint64_t)t2.getTimestamp() + FIXED_REPLY_DLY_TICKS;

      bool txOk = rawDelayedTransmit(txBuffer, 7, txTime);

      if (!txOk) {
        SPLN(F("[ERR] Delayed TX failed (HPDWARN or timeout)"));
        clearStatusAll();
        startReceiver();
        lastGoodMs = millis();
        return;
      }

      // ---- TX done — now read diagnostics (safe, not time-critical) ----
      DW1000Time t3;
      DW1000.getTransmitTimestamp(t3);
      int64_t actualReplyDelay = t3.getTimestamp() - t2.getTimestamp();

      RawDiag diag = readRxDiagnostics();

      clearStatusAll();

      // ---- BLE frame ----
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
      frame.reply_delay_lo = (int32_t)(actualReplyDelay & 0xFFFFFFFFLL);
      frame.reply_delay_hi = (uint8_t)((actualReplyDelay >> 32) & 0xFF);
      frame.flags          = 0x01;

      anchorChar.writeValue((uint8_t*)&frame, sizeof(AnchorFrame));

      lastGoodMs = millis();

      SP(F("A")); SP(DEVICE_ID);
      SP(F("<-T")); SP(tagId);
      SP(F(" #")); SP(rangeSeq);
      SP(F("  RX="));   SP2(diag.rx_power, 1);
      SP(F("  FP="));   SP2(diag.fp_power, 1);
      SP(F("  Q="));    SP2(diag.quality, 1);
      SP(F("  SN="));   SP(diag.std_noise);
      SP(F("  A1="));   SP(diag.fp_ampl1);
      SP(F("  A2="));   SP(diag.fp_ampl2);
      SP(F("  A3="));   SP(diag.fp_ampl3);
      SP(F("  PACC=")); SP(diag.rxpacc);
      SP(F("  RD="));   SP((long)actualReplyDelay);
      SPLN();
    }

    startReceiver();

  } else if (dataReady) {
    clearStatusAll();
    startReceiver();
  }

  delayMicroseconds(100);
}
