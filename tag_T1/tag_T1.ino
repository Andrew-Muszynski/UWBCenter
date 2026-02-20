// ============================================================
// UWB Tag (T1) — SS-TWR + BLE Telemetry
// Board: Arduino Nano 33 BLE Sense Lite + DWM1000 shield
//
// Performs Single-Sided TWR with anchor(s), then broadcasts
// raw timing + signal-quality metrics over BLE so that a
// Raspberry Pi (or any central) can collect everything.
//
// BLE device name format: T1, T2, T3 …  (set DEVICE_ID below)
// ============================================================

#include <SPI.h>
#include <DW1000.h>
#include <ArduinoBLE.h>

// ---------- identity ----------
#define DEVICE_ID   1                     // 1 → "T1", 2 → "T2", etc.
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

// ---------- ranging config ----------
#define RANGE_INTERVAL_MS  500            // how often to range (ms)
#define RX_TIMEOUT_MS       80            // per-response wait
#define TX_TIMEOUT_MS       50

// ---------- BLE UUIDs ----------
#define UWB_SERVICE_UUID  "19b10010-e8f2-537e-4f6c-d104768a1214"
#define TAG_CHAR_UUID     "19b10011-e8f2-537e-4f6c-d104768a1214"

// ---------- BLE data frame ----------
// 43 bytes total — well within the ~244-byte ATT payload the
// nRF52840 negotiates with a modern BLE central (e.g. Bleak).
//
// Field layout (little-endian, packed):
//   anchor_id      uint8    1   which anchor answered
//   seq            uint16   2   range sequence number
//   distance_m     float    4   computed distance (m)
//   round_trip_lo  int32    4   (T4-T1) lower 32 DW ticks
//   round_trip_hi  uint8    1   (T4-T1) upper  8 DW ticks
//   reply_delay_lo int32    4   (T3-T2) lower 32 DW ticks
//   reply_delay_hi uint8    1   (T3-T2) upper  8 DW ticks
//   rx_power       float    4   total RX power (dBm)
//   fp_power       float    4   first-path power (dBm)
//   quality        float    4   preamble quality (CIR/noise²)
//   std_noise      uint16   2   RX_FQUAL[0:1]  noise std-dev (raw)
//   fp_ampl1       uint16   2   RX_TIME[7:8]   1st-path amplitude (raw)
//   fp_ampl2       uint16   2   RX_FQUAL[2:3]  2nd-path amplitude (raw)
//   fp_ampl3       uint16   2   RX_FQUAL[4:5]  3rd-path amplitude (raw)
//   cir_power      uint16   2   RX_FQUAL[6:7]  CIR power (raw)
//   rxpacc         uint16   2   RX_FINFO[20:31] preamble accumulations
//   flags          uint8    1   bit0=range_ok, bit1=nlos_suspect
//   anchor_count   uint8    1   anchors heard
//                          ---
//                          43 bytes
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

// ---------- diagnostics bundle ----------
struct RxDiag {
  // Library-computed (dBm / quality metric)
  float    rx_power;
  float    fp_power;
  float    quality;
  // Raw register values — read from DW1000 before clearStatusAll()
  uint16_t std_noise;   // RX_FQUAL bytes 0-1
  uint16_t fp_ampl1;    // RX_TIME   bytes 7-8
  uint16_t fp_ampl2;    // RX_FQUAL bytes 2-3
  uint16_t fp_ampl3;    // RX_FQUAL bytes 4-5
  uint16_t cir_power;   // RX_FQUAL bytes 6-7
  uint16_t rxpacc;      // RX_FINFO  bits 20-31
};

// ---------- globals ----------
byte rxBuffer[20];
byte txBuffer[20];
uint16_t rangeSeq = 0;

BLEService        uwbService(UWB_SERVICE_UUID);
BLECharacteristic tagChar(TAG_CHAR_UUID, BLERead | BLENotify, sizeof(TagFrame));

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
    if (status[1] & 0x40) return true;   // RXFCG
    delayMicroseconds(100);
  }
  return false;
}

static inline bool waitForTxDone(uint16_t timeoutMs) {
  byte status[5];
  unsigned long start = millis();
  while ((millis() - start) < timeoutMs) {
    DW1000.readBytes(SYS_STATUS, NO_SUB, status, 5);
    if (status[0] & 0x80) return true;   // TXFRS
    delayMicroseconds(50);
  }
  return false;
}

// Read all RX diagnostics — MUST be called BEFORE clearStatusAll()
// so the DW1000 diagnostic registers (RX_FQUAL, RX_TIME, RX_FINFO)
// still hold values from the last received frame.
RxDiag readRxDiagnostics() {
  RxDiag d;

  // Library-computed metrics
  d.rx_power = DW1000.getReceivePower();
  d.fp_power = DW1000.getFirstPathPower();
  d.quality  = DW1000.getReceiveQuality();

  // RX_FQUAL register (0x12, 8 bytes):
  //   [0:1] STD_NOISE  [2:3] FP_AMPL2  [4:5] FP_AMPL3  [6:7] CIR_PWR
  byte fqual[8];
  DW1000.readBytes(0x12, NO_SUB, fqual, 8);
  d.std_noise = (uint16_t)fqual[0] | ((uint16_t)fqual[1] << 8);
  d.fp_ampl2  = (uint16_t)fqual[2] | ((uint16_t)fqual[3] << 8);
  d.fp_ampl3  = (uint16_t)fqual[4] | ((uint16_t)fqual[5] << 8);
  d.cir_power = (uint16_t)fqual[6] | ((uint16_t)fqual[7] << 8);

  // RX_TIME register (0x15): FP_AMPL1 is at bytes 7-8
  byte rxtime[9];
  DW1000.readBytes(0x15, NO_SUB, rxtime, 9);
  d.fp_ampl1 = (uint16_t)rxtime[7] | ((uint16_t)rxtime[8] << 8);

  // RX_FINFO register (0x10): RXPACC occupies bits 20-31
  byte rxfinfo[4];
  DW1000.readBytes(0x10, NO_SUB, rxfinfo, 4);
  uint32_t fi = (uint32_t)rxfinfo[0]        |
                ((uint32_t)rxfinfo[1] <<  8) |
                ((uint32_t)rxfinfo[2] << 16) |
                ((uint32_t)rxfinfo[3] << 24);
  d.rxpacc = (uint16_t)((fi >> 20) & 0xFFF);

  return d;
}

// ============================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println(F("=== UWB Tag + BLE  [" DEVICE_NAME "] ==="));

  DW1000.begin(PIN_IRQ, PIN_RST);
  DW1000.select(PIN_CS);

  DW1000.newConfiguration();
  DW1000.setDefaults();
  DW1000.setDeviceAddress(DEVICE_ID + 100);   // tags: 101, 102 …
  DW1000.setNetworkId(10);
  DW1000.enableMode(DW1000.MODE_LONGDATA_RANGE_LOWPOWER);
  DW1000.commitConfiguration();

  // Disable auto-sleep
  byte pmsc[4];
  DW1000.readBytes(0x36, 0x04, pmsc, 4);
  pmsc[1] &= ~0x18;
  DW1000.writeBytes(0x36, 0x04, pmsc, 4);

  // Disable DW1000 interrupts (we poll)
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
  BLE.addService(uwbService);

  TagFrame z = {};
  tagChar.writeValue((uint8_t*)&z, sizeof(TagFrame));

  BLE.advertise();
  Serial.println(F("BLE advertising. Ranging starts now.\n"));
}

// ============================================================
void loop() {
  BLE.poll();

  static unsigned long lastRange = 0;
  unsigned long now = millis();
  if (now - lastRange < RANGE_INTERVAL_MS) return;
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

  // ========== WAIT FOR RESPONSE #1 (T4) ==========
  DW1000.newReceive();
  DW1000.setDefaults();
  DW1000.receivePermanently(false);
  DW1000.startReceive();

  if (!waitForRxGood(RX_TIMEOUT_MS)) {
    Serial.print(F("T")); Serial.print(DEVICE_ID);
    Serial.print(F(" #")); Serial.print(rangeSeq);
    Serial.println(F("  RX TIMEOUT (resp1)"));
    clearStatusAll();
    return;
  }

  DW1000Time t4;
  DW1000.getReceiveTimestamp(t4);

  // Read ALL diagnostics now, before clearing status
  RxDiag diag = readRxDiagnostics();

  uint16_t len1 = DW1000.getDataLength();
  if (len1 > sizeof(rxBuffer)) len1 = sizeof(rxBuffer);
  DW1000.getData(rxBuffer, len1);

  uint8_t anchorId = 1;
  if (len1 >= 2) anchorId = rxBuffer[1];

  clearStatusAll();

  // ========== WAIT FOR RESPONSE #2 (reply delay T3-T2) ==========
  DW1000.newReceive();
  DW1000.setDefaults();
  DW1000.receivePermanently(false);
  DW1000.startReceive();

  if (!waitForRxGood(RX_TIMEOUT_MS)) {
    Serial.print(F("T")); Serial.print(DEVICE_ID);
    Serial.print(F(" #")); Serial.print(rangeSeq);
    Serial.println(F("  RX TIMEOUT (resp2)"));
    clearStatusAll();
    return;
  }

  uint16_t len2 = DW1000.getDataLength();
  if (len2 > sizeof(rxBuffer)) len2 = sizeof(rxBuffer);
  DW1000.getData(rxBuffer, len2);
  clearStatusAll();

  if (rxBuffer[0] != MSG_RESPONSE || len2 < 6) {
    Serial.print(F("T")); Serial.print(DEVICE_ID);
    Serial.print(F(" #")); Serial.print(rangeSeq);
    Serial.println(F("  BAD RESP2"));
    return;
  }

  // Extract full 40-bit reply delay from anchor (little-endian, 5 bytes)
  int64_t replyDelay = 0;
  replyDelay |= ((int64_t)rxBuffer[1] <<  0);
  replyDelay |= ((int64_t)rxBuffer[2] <<  8);
  replyDelay |= ((int64_t)rxBuffer[3] << 16);
  replyDelay |= ((int64_t)rxBuffer[4] << 24);
  replyDelay |= ((int64_t)rxBuffer[5] << 32);

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
  Serial.print(F("→A")); Serial.print(anchorId);
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
