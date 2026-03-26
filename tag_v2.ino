// ============================================================
// UWB Tag (T1) — SS-TWR + BLE Telemetry + Remote Commands
// Board: Arduino Nano 33 BLE Sense Lite + DWM1000 shield
//
// v2: Single-response protocol — anchor sends ONE packet with
//     both anchor_id and reply_delay, eliminating the resp2
//     timeout problem entirely.
//
// Protocol:
//   1. Tag sends POLL [0x01, tag_id]
//   2. Anchor sends RESPONSE [0x02, anchor_id, reply_delay(5 bytes)]
//   3. Tag computes ToF from round_trip and reply_delay
//
// BLE device name format: T1, T2, T3 ...  (set DEVICE_ID below)
// ============================================================

#include <SPI.h>
#include <DW1000.h>
#include <ArduinoBLE.h>

// ---------- identity ----------
#define DEVICE_ID   1                     // 1 -> "T1", 2 -> "T2", etc.
#define DEVICE_NAME "T1"                  // must match DEVICE_ID

// ---------- DW1000 wiring ----------
const uint8_t PIN_CS  = 10;
const uint8_t PIN_IRQ = 2;
const uint8_t PIN_RST = 3;

// ---------- TWR message types ----------
#define MSG_POLL     0x01
#define MSG_RESPONSE 0x02

// ---------- physics ----------
#define SPEED_OF_LIGHT 299702547.0
#define DW_TIME_UNITS  15.65e-12          // ~15.65 ps per tick

// ---------- ranging config (runtime-adjustable) ----------
uint16_t rangeIntervalMs = 500;           // how often to range (ms)
#define RX_TIMEOUT_MS       80            // wait for single response
#define TX_TIMEOUT_MS       50
uint16_t antennaDelay = 0;

// ---------- BLE UUIDs ----------
#define UWB_SERVICE_UUID  "19b10010-e8f2-537e-4f6c-d104768a1214"
#define TAG_CHAR_UUID     "19b10011-e8f2-537e-4f6c-d104768a1214"
#define CMD_CHAR_UUID     "19b10013-e8f2-537e-4f6c-d104768a1214"

// ---------- BLE data frame ----------
struct __attribute__((packed)) TagFrame {
  uint8_t  anchor_id;
  uint16_t seq;
  float    distance_m;
  int32_t  round_trip_lo;
  uint8_t  round_trip_hi;
  int32_t  reply_delay_lo;
  uint8_t  reply_delay_hi;
  float    rx_power;
  float    fp_power;
  float    quality;
  uint16_t std_noise;
  uint16_t fp_ampl1;
  uint16_t fp_ampl2;
  uint16_t fp_ampl3;
  uint16_t cir_power;
  uint16_t rxpacc;
  uint8_t  flags;
  uint8_t  anchor_count;
};

struct RxDiag {
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
uint16_t rangeSeq = 0;

BLEService        uwbService(UWB_SERVICE_UUID);
BLECharacteristic tagChar(TAG_CHAR_UUID, BLERead | BLENotify, sizeof(TagFrame));
BLECharacteristic cmdChar(CMD_CHAR_UUID, BLEWrite | BLERead, 32);

// ---------- helpers ----------

static inline void clearStatusAll() {
  byte clear[5] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
  DW1000.writeBytes(SYS_STATUS, NO_SUB, clear, 5);
}

static inline bool waitForRxGood(uint16_t timeoutMs) {
  byte status[5];
  unsigned long start = millis();
  while ((millis() - start) < timeoutMs) {
    DW1000.readBytes(SYS_STATUS, NO_SUB, status, 5);
    if (status[1] & 0x40) return true;     // RXDFR + RXFCG
    delayMicroseconds(100);
  }
  return false;
}

static inline bool waitForTxDone(uint16_t timeoutMs) {
  byte status[5];
  unsigned long start = millis();
  while ((millis() - start) < timeoutMs) {
    DW1000.readBytes(SYS_STATUS, NO_SUB, status, 5);
    if (status[0] & 0x80) return true;     // TXFRS
    delayMicroseconds(50);
  }
  return false;
}

RxDiag readRxDiagnostics() {
  RxDiag d;
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
  Serial.print(F("[CMD] Antenna delay set to: "));
  Serial.println(value);
}

void processCommand(const char* cmd) {
  Serial.print(F("[CMD] Received: "));
  Serial.println(cmd);

  if (strncmp(cmd, "AD:", 3) == 0) {
    uint16_t val = (uint16_t)atoi(cmd + 3);
    setAntennaDelay(val);
    char resp[32];
    snprintf(resp, sizeof(resp), "AD:%u OK", val);
    cmdChar.writeValue((uint8_t*)resp, strlen(resp));
  }
  else if (strncmp(cmd, "RI:", 3) == 0) {
    uint16_t val = (uint16_t)atoi(cmd + 3);
    if (val >= 50 && val <= 5000) {
      rangeIntervalMs = val;
      Serial.print(F("[CMD] Range interval set to: "));
      Serial.print(val);
      Serial.println(F(" ms"));
      char resp[32];
      snprintf(resp, sizeof(resp), "RI:%u OK", val);
      cmdChar.writeValue((uint8_t*)resp, strlen(resp));
    }
  }
  else if (strncmp(cmd, "ST", 2) == 0) {
    char resp[32];
    snprintf(resp, sizeof(resp), "AD:%u RI:%u", antennaDelay, rangeIntervalMs);
    cmdChar.writeValue((uint8_t*)resp, strlen(resp));
    Serial.print(F("[CMD] Status: "));
    Serial.println(resp);
  }
}

// ============================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println(F("=== UWB Tag v2 (single-resp) + BLE + CMD  [" DEVICE_NAME "] ==="));

  DW1000.begin(PIN_IRQ, PIN_RST);
  DW1000.select(PIN_CS);

  DW1000.newConfiguration();
  DW1000.setDefaults();
  DW1000.setDeviceAddress(DEVICE_ID + 100);
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
  uwbService.addCharacteristic(tagChar);
  uwbService.addCharacteristic(cmdChar);
  BLE.addService(uwbService);

  TagFrame z = {};
  tagChar.writeValue((uint8_t*)&z, sizeof(TagFrame));

  char initStatus[32];
  snprintf(initStatus, sizeof(initStatus), "AD:%u RI:%u", antennaDelay, rangeIntervalMs);
  cmdChar.writeValue((uint8_t*)initStatus, strlen(initStatus));

  BLE.advertise();
  Serial.println(F("BLE advertising. Ranging starts now.\n"));
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

  static unsigned long lastRange = 0;
  unsigned long now = millis();
  if (now - lastRange < rangeIntervalMs) return;
  lastRange = now;

  rangeSeq++;

  // ========== SEND POLL ==========
  txBuffer[0] = MSG_POLL;
  txBuffer[1] = DEVICE_ID;

  DW1000.newTransmit();
  DW1000.setDefaults();
  DW1000.setData(txBuffer, 2);
  DW1000.startTransmit();

  if (!waitForTxDone(TX_TIMEOUT_MS)) {
    Serial.print(F("T")); Serial.print(DEVICE_ID);
    Serial.print(F(" #")); Serial.print(rangeSeq);
    Serial.println(F("  TX TIMEOUT (POLL)"));
    clearStatusAll();
    return;
  }

  DW1000Time t1;
  DW1000.getTransmitTimestamp(t1);
  clearStatusAll();

  // ========== WAIT FOR SINGLE RESPONSE ==========
  // Anchor sends: [0x02, anchor_id, reply_delay(5 bytes)] = 7 bytes
  // The anchor uses delayed TX so its response arrives at a predictable time.
  // We give a generous timeout to account for the ~3ms fixed reply delay.
  DW1000.newReceive();
  DW1000.setDefaults();
  DW1000.receivePermanently(false);
  DW1000.startReceive();

  // Timeout needs to be > anchor's fixed reply delay (~3ms) + air time + margin
  // Using 80ms is very generous — could tighten to ~20ms if desired
  if (!waitForRxGood(RX_TIMEOUT_MS)) {
    Serial.print(F("T")); Serial.print(DEVICE_ID);
    Serial.print(F(" #")); Serial.print(rangeSeq);
    Serial.println(F("  RX TIMEOUT"));
    clearStatusAll();
    return;
  }

  DW1000Time t4;
  DW1000.getReceiveTimestamp(t4);

  RxDiag diag = readRxDiagnostics();

  uint16_t len1 = DW1000.getDataLength();
  if (len1 > sizeof(rxBuffer)) len1 = sizeof(rxBuffer);
  DW1000.getData(rxBuffer, len1);
  clearStatusAll();

  // ---- Validate response format ----
  // Expected: [MSG_RESPONSE, anchor_id, RD0, RD1, RD2, RD3, RD4] = 7 bytes
  if (rxBuffer[0] != MSG_RESPONSE || len1 < 7) {
    Serial.print(F("T")); Serial.print(DEVICE_ID);
    Serial.print(F(" #")); Serial.print(rangeSeq);
    Serial.print(F("  BAD RESP (type=0x"));
    Serial.print(rxBuffer[0], HEX);
    Serial.print(F(" len="));
    Serial.print(len1);
    Serial.println(F(")"));
    return;
  }

  uint8_t anchorId = rxBuffer[1];

  // Extract reply delay from bytes [2..6]
  int64_t replyDelay = 0;
  replyDelay |= ((int64_t)rxBuffer[2] <<  0);
  replyDelay |= ((int64_t)rxBuffer[3] <<  8);
  replyDelay |= ((int64_t)rxBuffer[4] << 16);
  replyDelay |= ((int64_t)rxBuffer[5] << 24);
  replyDelay |= ((int64_t)rxBuffer[6] << 32);

  // ========== CALCULATE DISTANCE ==========
  int64_t roundTrip = t4.getTimestamp() - t1.getTimestamp();
  if (roundTrip < 0) roundTrip += 0x10000000000LL;

  int64_t tof    = (roundTrip - replyDelay) / 2;
  double tofSec  = (double)tof * DW_TIME_UNITS;
  double dist    = tofSec * SPEED_OF_LIGHT;

  float fpRxRatio = diag.fp_power - diag.rx_power;
  bool  nlos      = (fpRxRatio < -6.0f);

  // ========== BUILD BLE FRAME ==========
  TagFrame frame;
  frame.anchor_id      = anchorId;
  frame.seq            = rangeSeq;
  frame.distance_m     = (float)dist;
  frame.round_trip_lo  = (int32_t)(roundTrip & 0xFFFFFFFFLL);
  frame.round_trip_hi  = (uint8_t)((roundTrip >> 32) & 0xFF);
  frame.reply_delay_lo = (int32_t)(replyDelay & 0xFFFFFFFFLL);
  frame.reply_delay_hi = (uint8_t)((replyDelay >> 32) & 0xFF);
  frame.rx_power       = diag.rx_power;
  frame.fp_power       = diag.fp_power;
  frame.quality        = diag.quality;
  frame.std_noise      = diag.std_noise;
  frame.fp_ampl1       = diag.fp_ampl1;
  frame.fp_ampl2       = diag.fp_ampl2;
  frame.fp_ampl3       = diag.fp_ampl3;
  frame.cir_power      = diag.cir_power;
  frame.rxpacc         = diag.rxpacc;
  frame.flags          = 0x01;
  if (nlos) frame.flags |= 0x02;
  frame.anchor_count   = 1;

  tagChar.writeValue((uint8_t*)&frame, sizeof(TagFrame));

  // ========== SERIAL DEBUG ==========
  Serial.print(F("T")); Serial.print(DEVICE_ID);
  Serial.print(F("->A")); Serial.print(anchorId);
  Serial.print(F(" #")); Serial.print(rangeSeq);
  Serial.print(F("  d="));  Serial.print(dist, 3); Serial.print(F("m"));
  Serial.print(F("  RX="));  Serial.print(diag.rx_power, 1);
  Serial.print(F("  FP="));  Serial.print(diag.fp_power, 1);
  Serial.print(F("  Q="));   Serial.print(diag.quality, 1);
  Serial.print(F("  SN="));  Serial.print(diag.std_noise);
  Serial.print(F("  A1="));  Serial.print(diag.fp_ampl1);
  Serial.print(F("  A2="));  Serial.print(diag.fp_ampl2);
  Serial.print(F("  A3="));  Serial.print(diag.fp_ampl3);
  Serial.print(F("  PACC=")); Serial.print(diag.rxpacc);
  Serial.print(F("  RT="));  Serial.print((long)roundTrip);
  Serial.print(F("  RD="));  Serial.print((long)replyDelay);
  if (nlos) Serial.print(F("  NLOS?"));
  Serial.println();
}
