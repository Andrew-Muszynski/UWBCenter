# UWB ML Ranging Correction System — Architecture

## Overview

This system corrects systematic ranging errors in DWM1000 Ultra-Wideband (UWB) hardware by applying machine learning models trained on labelled signal-quality data. It spans three tiers: embedded firmware on Arduino Nano 33 BLE boards, an edge gateway on a Raspberry Pi 5, and ML client applications that train and apply correction models in real time.

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TIER 1 — EMBEDDED FIRMWARE                       │
│                                                                         │
│   ┌──────────────────────┐         SS-TWR          ┌─────────────────┐  │
│   │  Tag  (T1)           │ ──── POLL [0x01] ─────▶ │  Anchor  (A1)   │  │
│   │  Arduino Nano 33 BLE │ ◀─── RESP [0x02] ────── │  Arduino Nano   │  │
│   │  + DWM1000 shield    │    (anchor_id + delay)   │  33 BLE +       │  │
│   │                      │                          │  DWM1000 shield │  │
│   │  Computes:           │                          │                 │  │
│   │  • ToF distance      │                          │  Computes:      │  │
│   │  • RX diagnostics    │                          │  • RX diags     │  │
│   │  • NLOS detection    │                          │  • Reply delay  │  │
│   │                      │                          │                 │  │
│   │  BLE: TagFrame       │                          │  BLE: AnchorFrm │  │
│   │       (43 bytes)     │                          │       (33 bytes)│  │
│   └──────────┬───────────┘                          └────────┬────────┘  │
│              │ BLE Notify                                    │ BLE       │
│              │ TAG_CHAR_UUID                                 │ Notify    │
│              │                                               │ ANCHOR_   │
│              │                                               │ CHAR_UUID │
└──────────────┼───────────────────────────────────────────────┼──────────┘
               │                                               │
               ▼                                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     TIER 2 — EDGE GATEWAY (Mac OS running)              │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │                     uwb_dashboard.py                              │ │
│   │                                                                   │ │
│   │  BLE Layer (asyncio + Bleak)                                      │ │
│   │  ├─ BleakScanner: discover T1..T10, A1..A10 every 15s            │ │
│   │  ├─ handle_tag():  subscribe TAG_CHAR → unpack TagFrame           │ │
│   │  ├─ handle_anchor(): subscribe ANCHOR_CHAR → unpack AnchorFrame   │ │
│   │  ├─ Auto-reconnect with address refresh on disconnect             │ │
│   │  └─ Command queue: pending_commands → write CMD_CHAR_UUID         │ │
│   │                                                                   │ │
│   │  Data Layer                                                       │ │
│   │  ├─ CSV Logger: tag_data_*.csv, anchor_data_*.csv                 │ │
│   │  ├─ Logs routed to AD_<value>/ subfolder per antenna delay        │ │
│   │  ├─ In-memory history: deque(maxlen=200) per device               │ │
│   │  └─ Labelled collection → ble_cal_dataset.csv                     │ │
│   │                                                                   │ │
│   │  Flask API (port 5050)                                            │ │
│   │  ├─ GET  /api/state          → full device snapshot (JSON)        │ │
│   │  ├─ POST /api/command        → send BLE command to one device     │ │
│   │  ├─ POST /api/set_all_delay  → broadcast antenna delay            │ │
│   │  ├─ POST /api/collection/start → start labelled data collection   │ │
│   │  ├─ POST /api/collection/stop  → stop & save to cal dataset       │ │
│   │  ├─ GET  /api/collection/status → live collection stats           │ │
│   │  └─ GET  /api/collection/dataset_info → cal dataset summary       │ │
│   │                                                                   │ │
│   │  Web Dashboard (dashboard.html)                                   │ │
│   │  └─ Real-time device cards, signal charts, command controls       │ │
│   └───────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────┬──────────────────────────────┘
                                           │ HTTP /api/state
                                           │ (polled every 0.5s)
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        TIER 3 — ML CLIENTS                              │
│                                                                         │
│   ┌─────────────────────────────────┐  ┌──────────────────────────────┐ │
│   │  uwb_ble_calibration.py        │  │  UWB_calibration_ml.py       │ │
│   │  (BLE-based, Tkinter GUI)      │  │  (Serial-based, Tkinter GUI) │ │
│   │                                 │  │                              │ │
│   │  Tabs:                          │  │  Tabs:                       │ │
│   │  1. Collect — live BLE → label  │  │  1. Data Collection (serial) │ │
│   │  2. Import  — bulk CSV import   │  │  2. Dataset management       │ │
│   │  3. Dataset — view & manage     │  │  3. ML Training              │ │
│   │  4. Train   — fit models        │  │  4. Live Correction          │ │
│   │  5. Infer   — real-time correct │  │                              │ │
│   │                                 │  │  Connects directly to        │ │
│   │  Polls dashboard /api/state     │  │  Arduino over USB serial     │ │
│   │  via BLEPoller (requests lib)   │  │                              │ │
│   └─────────────────────────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Embedded Firmware (Tier 1)

#### Tag Firmware (`tag_v2.ino` / `tag_T1/tag_T1.ino`)

- **Board**: Arduino Nano 33 BLE Sense Lite + DWM1000 UWB shield
- **Protocol**: Single-Sided Two-Way Ranging (SS-TWR)
  1. Tag sends `POLL [0x01, tag_id]`
  2. Anchor replies `RESPONSE [0x02, anchor_id, reply_delay(5 bytes)]`
  3. Tag computes Time-of-Flight: `ToF = (round_trip - reply_delay) / 2`
  4. Distance: `d = ToF * 15.65e-12 * 299702547.0`
- **BLE Output**: `TagFrame` struct (43 bytes, packed) containing:
  - `anchor_id`, `seq`, `distance_m`
  - `round_trip` (40-bit), `reply_delay` (40-bit)
  - `rx_power`, `fp_power`, `quality` (floats)
  - `std_noise`, `fp_ampl1`, `fp_ampl2`, `fp_ampl3`, `cir_power`, `rxpacc` (uint16)
  - `flags` (NLOS bit), `anchor_count`
- **BLE Commands** (via `CMD_CHAR_UUID`):
  - `AD:<value>` — set antenna delay register (0–65535)
  - `RI:<value>` — set range interval in ms (50–5000)
  - `ST` — query current settings
- **Configurable at runtime**: antenna delay, range interval (default 500 ms)

#### Anchor Firmware (`anchor_v4.ino` / `anchor_A1/anchor_A1.ino`)

- **Protocol**: Listens for POLL, responds with raw-register delayed TX (bypasses DW1000 library overhead to avoid HPDWARN)
- **Fixed Reply Delay**: 10 ms (`638,977,636` DW1000 ticks)
- **BLE Output**: `AnchorFrame` struct (33 bytes, packed) containing:
  - `tag_id`, `seq`
  - `rx_power`, `fp_power`, `quality` (floats)
  - `std_noise`, `fp_ampl1`, `fp_ampl2`, `fp_ampl3`, `cir_power`, `rxpacc` (uint16)
  - `reply_delay` (40-bit), `flags`
- **BLE Commands**: `AD:<value>`, `ST`
- **Watchdog**: 5-second timeout triggers DWM soft-reset if no successful ranging

#### BLE Service Definition

| UUID | Characteristic | Direction |
|------|---------------|-----------|
| `19b10010-e8f2-537e-4f6c-d104768a1214` | UWB Service | — |
| `19b10011-e8f2-537e-4f6c-d104768a1214` | Tag Data | Read/Notify |
| `19b10012-e8f2-537e-4f6c-d104768a1214` | Anchor Data | Read/Notify |
| `19b10013-e8f2-537e-4f6c-d104768a1214` | Command | Write/Read |

---

### 2. Edge Gateway — `uwb_dashboard.py` (Tier 2)

Runs on Raspberry Pi 5. Combines an asynchronous BLE client with a Flask web server in a single process.

#### BLE Layer

- **Scanner**: `BleakScanner.discover()` runs every 15 seconds, looking for devices named `T1`–`T10` and `A1`–`A10`
- **Connection management**: Each discovered device gets a persistent `asyncio.Task` that connects, subscribes to notifications, and reconnects on failure with a 3-second backoff
- **Address refresh**: On disconnect, re-scans to handle Arduino reboot (new BLE address)
- **Struct unpacking**:
  - TagFrame: `<BHfiBiBfffHHHHHHBB` (43 bytes)
  - AnchorFrame: `<BHfffHHHHHHiBB` (33 bytes)
- **Derived fields**: `fp_rx_ratio = fp_power - rx_power`, `nlos_suspect = (fp_rx_ratio < -6.0)`

#### Data Layer

- **CSV logging**: Every packet is written to timestamped CSV files (`tag_data_YYYYMMDD_HHMMSS.csv`, `anchor_data_...`)
- **Antenna delay routing**: Logs are stored in `logs/AD_<value>/` subfolders (e.g., `logs/AD_16700/`, `logs/AD_22750/`). When antenna delay changes, CSVs are rotated to the new subfolder
- **In-memory state**: `deque(maxlen=200)` per device for the API
- **Labelled collection**: Dashboard API supports starting a labelled data collection session with known `true_dist_m` and `angle_deg`. Collected samples are appended to `ble_cal_dataset.csv` in the appropriate AD subfolder
- **Auto-stop**: Collection automatically stops when `target_samples` is reached

#### Flask API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the live web dashboard |
| `/api/state` | GET | JSON snapshot of all devices (last 50 packets each) |
| `/api/command` | POST | Send a BLE command to a specific device |
| `/api/set_all_delay` | POST | Broadcast antenna delay to all connected devices |
| `/api/collection/start` | POST | Start labelled data collection |
| `/api/collection/stop` | POST | Stop collection, save to cal dataset |
| `/api/collection/status` | GET | Live collection stats |
| `/api/collection/dataset_info` | GET | Summary of saved calibration dataset |

#### Web Dashboard (`templates/dashboard.html`)

- Dark-themed single-page app
- Real-time device cards with connection status indicators
- Signal quality charts and ranging data
- Command controls for antenna delay and range interval
- Polls `/api/state` at ~1 Hz

---

### 3. ML Clients (Tier 3)

#### BLE-Based Client — `uwb_ble_calibration.py`

Primary ML tool. Connects to the dashboard API over HTTP (not directly to BLE).

**Architecture**:
- `BLEPoller` thread polls `/api/state` every 0.5 seconds
- Deduplicates packets by `seq` number (only processes new packets)
- Dispatches to registered callbacks for collection and inference

**5-Tab Tkinter GUI**:

1. **Collect** — Start/stop labelled sessions with known distance and angle. Live mini-plot shows distance readings and true reference line
2. **Import Logs** — Bulk-import existing `tag_data_*.csv` files with user-supplied labels (distance and angle per file)
3. **Dataset** — View, export, or clear accumulated calibration data. Shows per-distance/angle summary statistics (mean, std, MAE)
4. **Train** — Configure and train ML models (see ML Pipeline below)
5. **Live Inference** — Apply trained models in real time with 2D polar position map, time series, and error metrics

**Feature Engineering** (`engineer()` function):

| Feature | Formula | Purpose |
|---------|---------|---------|
| `ampl1_ratio` | `fp_ampl1 / mean(fp_ampl2, fp_ampl3)` | First-path dominance indicator |
| `cir_norm` | `cir_power / rxpacc` | Normalised channel impulse response |
| `ampl_spread` | `abs(fp_ampl2 - fp_ampl3)` | Multipath spread indicator |

**Raw Features** (from each tag BLE packet):
`distance_m`, `rx_power`, `fp_power`, `fp_rx_ratio`, `quality`, `std_noise`, `fp_ampl1`, `fp_ampl2`, `fp_ampl3`, `cir_power`, `rxpacc`

**ML Pipeline**:

- **Distance Corrector** (regression): Predicts `true_dist_m` from selected features
  - Algorithms: `GradientBoostingRegressor`, `RandomForestRegressor`, `Ridge + PolynomialFeatures(degree=3)`
  - Default features: `distance_m`, `fp_rx_ratio`, `quality`, `ampl1_ratio`, `cir_norm`, `rx_power`
  - Wrapped in sklearn `Pipeline` with `StandardScaler`
  - GBR config: 400 estimators, max_depth=4, learning_rate=0.05, subsample=0.8
  - Evaluation: MAE, RMSE, R², k-fold cross-validation, per-distance breakdown, feature importance
- **Angle Classifier** (optional): Predicts tag orientation angle from signal features
  - Algorithm: `RandomForestClassifier` (400 estimators, max_depth=12)
  - Default features: `fp_rx_ratio`, `quality`, `ampl1_ratio`, `ampl_spread`, `std_noise`, `rx_power`
  - Classes: discrete angles (0, 45, 90, 135, 180, 225, 270, 315 degrees)

**Model Persistence**: Models are serialized via `pickle` as a dict containing `dist_model`, `dist_feats`, `angle_model`, `angle_feats`, and `meta`.

**Live Inference Flow**:
1. BLEPoller receives new packet from `/api/state`
2. Raw features extracted, `engineer()` adds derived features
3. `safe_build_X()` builds feature matrix (NaN → column median)
4. Distance model predicts corrected distance
5. Angle model (if loaded) predicts orientation
6. UI updates: raw vs corrected distance, error, estimated angle, polar map with position trail

#### Serial-Based Client — `UWB_calibration_ml.py`

Alternative tool for direct USB serial connection to an Arduino (no Raspberry Pi needed).

**4-Tab Tkinter GUI**:
1. **Data Collection** — Connect to serial port, send commands (`start N`, `stop`, `delay XXXXX`, `distance X.XX`)
2. **Dataset** — Manage accumulated samples
3. **ML Training** — Same algorithm options (Polynomial, Ridge Poly, GBR, Random Forest)
4. **Live Correction** — Apply model to serial stream in real time

**CSV Format** (from Arduino serial output):
`sample, millis, distance_m, round_trip_ticks, reply_delay_ticks, tof_ticks, rx_power_dBm, fp_power_dBm, quality, temp_C, pressure_hPa`

---

### 4. Data Flow

```
Tag DWM1000 ─── SS-TWR ───▶ Anchor DWM1000
     │                            │
     │ BLE TagFrame (43B)         │ BLE AnchorFrame (33B)
     ▼                            ▼
   Raspberry Pi 5 (uwb_dashboard.py)
     │
     ├──▶ CSV logs (logs/AD_<delay>/tag_data_*.csv)
     ├──▶ ble_cal_dataset.csv (labelled data)
     ├──▶ Flask /api/state (JSON, polled by clients)
     │
     ▼
   ML Client (uwb_ble_calibration.py)
     │
     ├──▶ BLEPoller: dedup by seq, dispatch new packets
     ├──▶ Collect: label with true_dist + angle → dataset
     ├──▶ Train: sklearn Pipeline (GBR/RF/Ridge) → .pkl
     └──▶ Infer: real-time corrected distance + angle
```

---

### 5. Directory Structure

```
ML_Project/
├── anchor_v4.ino              # Anchor firmware (v4, raw delayed TX)
├── tag_v2.ino                 # Tag firmware (v2, single-response)
├── anchor_A1/
│   └── anchor_A1.ino          # Device-specific anchor sketch
├── tag_T1/
│   └── tag_T1.ino             # Device-specific tag sketch
├── uwb_dashboard.py           # Edge gateway: BLE + Flask + CSV logging
├── uwb_ble_calibration.py     # ML client: BLE-based collection, training, inference
├── UWB_calibration_ml.py      # ML client: Serial-based (direct USB to Arduino)
├── generate_design_doc.py     # PDF generator for the ML system design document
├── templates/
│   └── dashboard.html         # Web dashboard frontend
└── logs/
    ├── AD_16700/              # Logs for antenna delay 16700
    │   ├── tag_data_*.csv
    │   ├── anchor_data_*.csv
    │   └── ble_cal_dataset.csv
    ├── AD_22750/              # Logs for antenna delay 22750
    │   ├── tag_data_*.csv
    │   ├── anchor_data_*.csv
    │   └── ble_cal_dataset.csv
    └── AD_unknown/            # Logs before antenna delay is known
        ├── tag_data_*.csv
        └── anchor_data_*.csv
```

---

### 6. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| SS-TWR (not DS-TWR) | Simpler protocol, one round-trip. Systematic error is corrected by ML instead of a second exchange |
| Fixed 10 ms reply delay | Generous margin avoids HPDWARN (half-period delay warning) on the DW1000 |
| Raw register delayed TX on anchor | Bypasses DW1000 library's `newTransmit()`/`setDefaults()` overhead |
| BLE for telemetry (not serial) | Enables wireless multi-device monitoring from a single Pi gateway |
| Antenna delay subfolder routing | Isolates datasets per calibration configuration for reproducibility |
| GBR as default distance corrector | Handles non-linear error patterns; regularized via learning_rate + subsample |
| Feature engineering (ampl1_ratio, cir_norm, ampl_spread) | Captures first-path dominance and multipath effects that raw features miss |
| BLEPoller deduplication by seq | Prevents double-counting when polling overlapping API responses |
| sklearn Pipeline with StandardScaler | Ensures feature normalization is applied consistently during train and inference |
| Pickle for model serialization | Fast save/load for prototyping; models are local-only |

---

### 7. Communication Protocols

#### UWB (SS-TWR over DWM1000)

- Mode: `MODE_LONGDATA_RANGE_LOWPOWER`
- Network ID: 10
- Poll frame: `[0x01, tag_id]` (2 bytes)
- Response frame: `[0x02, anchor_id, reply_delay_byte0..4]` (7 bytes)
- Timing: poll → 10 ms fixed delay → response
- Distance calculation: `(round_trip - reply_delay) / 2 * 15.65ps * c`

#### BLE

- Service UUID: `19b10010-e8f2-537e-4f6c-d104768a1214`
- Characteristics: Tag data (notify), Anchor data (notify), Command (write/read)
- Struct packing: little-endian, no padding (`__attribute__((packed))`)
- Command protocol: ASCII strings (e.g., `AD:16700`, `RI:500`, `ST`)
- Response protocol: ASCII strings (e.g., `AD:16700 OK`, `AD:16700 RI:500`)

#### HTTP (Dashboard ↔ ML Client)

- Transport: HTTP/1.1, JSON payloads
- Poll rate: 500 ms (configurable via `POLL_INTERVAL`)
- State endpoint returns last 50 packets per device
- Command endpoint queues commands for the BLE thread to process asynchronously
