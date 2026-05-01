// ============================================================
// UWB Tag (T1) — SS-TWR Initiator + BLE Telemetry
// Board: Arduino Nano 33 BLE Sense Lite + DWM1000 shield
//
// Built directly on the working SS_RANGING_TAG_33 protocol:
//   1. Tag  sends POLL  [0x01, tag_id]
//   2. Tag  receives RESP1 [0x02, anchor_id]  <- t4_first captured here
//   3. Tag  receives RESP2 [0x02, rd0..rd4]   <- reply delay bytes
//   4. ToF  = (t4_first - t1 - replyDelay) / 2
//
// BLE frame (43 bytes) matches Python "<BHfiBiBfffHHHHHHBB"
// Device name "T1" is discovered by uwb_dashboard.py
// ============================================================

#include <SPI.h>
#include <DW1000.h>
#include <ArduinoBLE.h>

#define DEVICE_ID   1
#define DEVICE_NAME "T1"

const uint8_t PIN_CS  = 10;
const uint8_t PIN_IRQ = 2;
const uint8_t PIN_RST = 3;

#define MSG_POLL       0x01
#define MSG_RESPONSE   0x02
#define SPEED_OF_LIGHT 299702547.0
#define DW_TIME_UNITS  15.65e-12

uint16_t rangeIntervalMs = 1000;   // ms between polls; tune via BLE "RI:NNNN"
uint16_t antennaDelay    = 16384;  // tune via BLE "AD:NNNN"

// BLE UUIDs — must match uwb_dashboard.py exactly
#define UWB_SERVICE_UUID "19b10010-e8f2-537e-4f6c-d104768a1214"
#define TAG_CHAR_UUID    "19b10011-e8f2-537e-4f6c-d104768a1214"
#define CMD_CHAR_UUID    "19b10013-e8f2-537e-4f6c-d104768a1214"

// BLE data frame — layout matches Python: "<BHfiBiBfffHHHHHHBB" = 43 bytes
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
  uint8_t  flags;        // bit0=valid, bit1=NLOS suspect
  uint8_t  anchor_count;
};

byte     rxBuffer[20];
byte     txBuffer[20];
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
  byte st[5];
  unsigned long t = millis();
  while (millis() - t < timeoutMs) {
    DW1000.readBytes(SYS_STATUS, NO_SUB, st, 5);
    if ((st[1] & 0x20) && (st[1] & 0x40)) return true;  // RXDFR && RXFCG
    delayMicroseconds(100);
  }
  return false;
}

static inline bool waitForTxDone(uint16_t timeoutMs) {
  byte st[5];
  unsigned long t = millis();
  while (millis() - t < timeoutMs) {
    DW1000.readBytes(SYS_STATUS, NO_SUB, st, 5);
    if (st[0] & 0x80) return true;  // TXFRS
    delayMicroseconds(50);
  }
  return false;
}

void applyAntennaDelay() {
  byte buf[2] = { (byte)(antennaDelay & 0xFF), (byte)((antennaDelay >> 8) & 0xFF) };
  DW1000.writeBytes(0x18, 0x00, buf, 2);
  DW1000.writeBytes(0x2E, 0x1804, buf, 2);
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
  } else if (strncmp(cmd, "RI:", 3) == 0) {
    uint16_t v = (uint16_t)atoi(cmd + 3);
    if (v >= 200 && v <= 5000) {
      rangeIntervalMs = v;
      char resp[32];
      snprintf(resp, sizeof(resp), "RI:%u OK", v);
      cmdChar.writeValue((uint8_t*)resp, strlen(resp));
      Serial.print("[CMD] RI="); Serial.println(v);
    }
  } else if (strncmp(cmd, "ST", 2) == 0) {
    char resp[32];
    snprintf(resp, sizeof(resp), "AD:%u RI:%u", antennaDelay, rangeIntervalMs);
    cmdChar.writeValue((uint8_t*)resp, strlen(resp));
  }
}

// ============================================================
void setup() {
  Serial.begin(115200);
  // wait up to 2s for serial monitor; proceed regardless
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 2000) { delay(10); }

  Serial.println("=== UWB Tag v3 + BLE [" DEVICE_NAME "] ===");

  DW1000.begin(PIN_IRQ, PIN_RST);
  DW1000.select(PIN_CS);

  // SPI sanity check — must read 0xDECA0130
  byte devId[4];
  DW1000.readBytes(0x00, 0x00, devId, 4);
  Serial.print("DEVID: 0x");
  for (int i = 3; i >= 0; i--) {
    if (devId[i] < 0x10) Serial.print("0");
    Serial.print(devId[i], HEX);
  }
  Serial.println();

  DW1000.newConfiguration();
  DW1000.setDefaults();
  DW1000.setDeviceAddress(DEVICE_ID + 100);  // tag addresses offset from anchors
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

  if (!BLE.begin()) {
    Serial.println("BLE init failed!");
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
  Serial.println("BLE advertising. Ranging starts in 2s...\n");
  delay(2000);
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

  static unsigned long lastRange = 0;
  if (millis() - lastRange < rangeIntervalMs) return;
  lastRange = millis();

  rangeSeq++;

  // ========== SEND POLL ==========
  txBuffer[0] = MSG_POLL;
  txBuffer[1] = DEVICE_ID;

  DW1000.idle();
  DW1000.newTransmit();
  DW1000.setData(txBuffer, 2);
  DW1000.startTransmit();

  if (!waitForTxDone(50)) {
    Serial.print("T"); Serial.print(DEVICE_ID);
    Serial.print(" #"); Serial.print(rangeSeq);
    Serial.println("  TX TIMEOUT (POLL)");
    clearStatusAll();
    DW1000.idle();
    return;
  }

  DW1000Time t1;
  DW1000.getTransmitTimestamp(t1);
  clearStatusAll();

  // ========== WAIT FOR RESP1 (carries anchor_id, timestamps the exchange) ==========
  DW1000.newReceive();
  DW1000.receivePermanently(false);
  DW1000.startReceive();

  if (!waitForRxGood(100)) {
    Serial.print("T"); Serial.print(DEVICE_ID);
    Serial.print(" #"); Serial.print(rangeSeq);
    Serial.println("  RX TIMEOUT (RESP1)");
    clearStatusAll();
    DW1000.idle();
    return;
  }

  DW1000Time t4_first;
  DW1000.getReceiveTimestamp(t4_first);

  // Read diagnostics from RESP1 (the useful signal metrics)
  float rxp, fpp, qual;
  uint16_t sn, a1, a2, a3, cp, pacc;
  readDiagnostics(rxp, fpp, qual, sn, a1, a2, a3, cp, pacc);

  uint16_t len1 = DW1000.getDataLength();
  if (len1 > sizeof(rxBuffer)) len1 = sizeof(rxBuffer);
  DW1000.getData(rxBuffer, len1);
  clearStatusAll();

  uint8_t anchorId = (len1 >= 2 && rxBuffer[0] == MSG_RESPONSE) ? rxBuffer[1] : 0;

  // ========== WAIT FOR RESP2 (carries reply delay for ToF calculation) ==========
  DW1000.newReceive();
  DW1000.receivePermanently(false);
  DW1000.startReceive();

  if (!waitForRxGood(100)) {
    Serial.print("T"); Serial.print(DEVICE_ID);
    Serial.print(" #"); Serial.print(rangeSeq);
    Serial.println("  RX TIMEOUT (RESP2)");
    clearStatusAll();
    DW1000.idle();
    return;
  }

  uint16_t len2 = DW1000.getDataLength();
  if (len2 > sizeof(rxBuffer)) len2 = sizeof(rxBuffer);
  DW1000.getData(rxBuffer, len2);
  clearStatusAll();

  if (rxBuffer[0] != MSG_RESPONSE || len2 < 6) {
    Serial.print("T"); Serial.print(DEVICE_ID);
    Serial.print(" #"); Serial.print(rangeSeq);
    Serial.print("  BAD RESP2 (0x"); Serial.print(rxBuffer[0], HEX);
    Serial.print(" len="); Serial.print(len2); Serial.println(")");
    return;
  }

  // ========== COMPUTE DISTANCE ==========
  int64_t replyDelay = 0;
  replyDelay |= ((int64_t)rxBuffer[1] <<  0);
  replyDelay |= ((int64_t)rxBuffer[2] <<  8);
  replyDelay |= ((int64_t)rxBuffer[3] << 16);
  replyDelay |= ((int64_t)rxBuffer[4] << 24);
  replyDelay |= ((int64_t)rxBuffer[5] << 32);

  int64_t roundTrip = t4_first.getTimestamp() - t1.getTimestamp();
  if (roundTrip < 0) roundTrip += 0x10000000000LL;

  int64_t tof   = (roundTrip - replyDelay) / 2;
  double tofSec = (double)tof * DW_TIME_UNITS;
  double dist   = tofSec * SPEED_OF_LIGHT;

  float fpRxRatio = fpp - rxp;
  bool  nlos      = (fpRxRatio < -6.0f);

  // ========== BUILD AND SEND BLE FRAME ==========
  TagFrame frame;
  frame.anchor_id      = anchorId;
  frame.seq            = rangeSeq;
  frame.distance_m     = (float)dist;
  frame.round_trip_lo  = (int32_t)(roundTrip & 0xFFFFFFFFLL);
  frame.round_trip_hi  = (uint8_t)((roundTrip >> 32) & 0xFF);
  frame.reply_delay_lo = (int32_t)(replyDelay & 0xFFFFFFFFLL);
  frame.reply_delay_hi = (uint8_t)((replyDelay >> 32) & 0xFF);
  frame.rx_power       = rxp;
  frame.fp_power       = fpp;
  frame.quality        = qual;
  frame.std_noise      = sn;
  frame.fp_ampl1       = a1;
  frame.fp_ampl2       = a2;
  frame.fp_ampl3       = a3;
  frame.cir_power      = cp;
  frame.rxpacc         = pacc;
  frame.flags          = 0x01 | (nlos ? 0x02 : 0x00);
  frame.anchor_count   = 1;
  tagChar.writeValue((uint8_t*)&frame, sizeof(TagFrame));

  Serial.print("T"); Serial.print(DEVICE_ID);
  Serial.print(" -> A"); Serial.print(anchorId);
  Serial.print("  #"); Serial.print(rangeSeq);
  Serial.print("  d="); Serial.print(dist, 3); Serial.print("m");
  Serial.print("  RX="); Serial.print(rxp, 1);
  Serial.print("  FP="); Serial.print(fpp, 1);
  Serial.print("  Q=");  Serial.print(qual, 1);
  if (nlos) Serial.print("  NLOS?");
  Serial.println();
}