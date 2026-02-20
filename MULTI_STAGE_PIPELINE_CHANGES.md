# Multi-Stage ML Pipeline Updates

## Overview

Both `UWB_calibration_ml.py` (serial) and `uwb_ble_calibration.py` (BLE) have been updated
to implement a layered ML architecture that separates the problem into distinct stages:

1. **Stage 1: Channel Classification** - Classify each measurement as LOS / Marginal / NLOS
2. **Stage 2: Distance Correction** - Regression with channel condition as an input feature
3. **Stage 3: Confidence Scoring** - Per-measurement quality weight for downstream use

## New Functions (both files)

### `classify_channel()` / `classify_channel_ble()`
Classifies channel condition based on DW1000 User Manual Section 4.7 thresholds:
- **LOS** (class 0): power difference < 6 dB (first path carries most energy)
- **Marginal** (class 1): power difference 6-10 dB
- **NLOS** (class 2): power difference > 10 dB (heavy multipath/obstruction)

The serial version uses `power_diff_dB = rx_power_dBm - fp_power_dBm` (positive values).
The BLE version uses `fp_rx_ratio = fp_power - rx_power` (negative values, same physics).

### `compute_confidence()` / `compute_confidence_ble()`
Returns a 0-1 confidence score combining:
- **Quality** (60% weight): sigmoid-normalized preamble accumulation quality
- **Power difference** (40% weight): how close to LOS the channel is
- **Optional noise/std penalty**: penalizes high-variance or high-noise measurements

## New Features Added to Feature Selection

| Feature | Description | Use |
|---------|------------|-----|
| `channel_condition` | 0=LOS, 1=Marginal, 2=NLOS | Tells the regressor which correction curve to apply |
| `confidence` | 0.0 - 1.0 measurement quality | Weighting for trilateration / filtering |

Both are now **on by default** in the feature selection checkboxes.

## Training Pipeline Changes

### Multi-Stage Pipeline (checkbox, default ON)
When enabled, training runs in stages:
1. Trains an NLOS Random Forest classifier on signal diagnostics
2. Reports per-channel-condition error statistics (critical for diagnosing your setup)
3. Feeds `channel_condition` as an input feature to the distance regressor
4. Reports confidence distribution and low-confidence warnings

### LODO Cross-Validation (checkbox, default OFF)
When enabled and the dataset has 2+ distinct true distances:
- Uses `LeaveOneGroupOut` where groups = `true_distance_m`
- Each fold holds out all samples at one distance, trains on the rest
- Tests the model's ability to interpolate/extrapolate to unseen distances
- Reports per-fold MAE (e.g., "Held out 1.00m: MAE = 4.2 cm")

When dataset has only 1 distance, falls back to standard k-fold with a note.

### Improved Plotting (serial version)
The training results now show a 2x3 grid:
1. Raw vs Corrected scatter
2. Error histogram
3. Error vs Distance
4. Residual plot
5. **Error colored by Channel Condition** (LOS=green, Marginal=orange, NLOS=red)
6. **Confidence vs Absolute Error** (shows if confidence predicts error magnitude)

## Live Correction Changes

### Serial version (`_update_live`)
Now displays alongside each corrected reading:
- Channel condition label (LOS / MARGINAL / NLOS)
- Confidence score (0.00 - 1.00)
- Color-coded by channel condition (green/orange/red)

### BLE version (`_update_inference`)
The error display now includes channel condition and confidence inline:
```
+0.032 m  [LOS  conf=0.78]
```

## Model Save/Load
Both versions now persist the NLOS classifier alongside the distance model in the `.pkl` file.
Loading an older `.pkl` without the NLOS model gracefully falls back to single-stage.

## Data Collection
The serial version now computes `channel_condition` and `confidence` on every incoming
sample during collection, so these features are available in the dataset immediately.

The BLE version computes them in `engineer()`, which runs on import and during inference.

## Key Finding from Your Existing Data
Running the new classification on `ble_cal_dataset.csv`:
- **99.1% of samples are NLOS** (fp_rx_ratio around -14 to -17 dB)
- Only 1 sample classified as LOS, 8 as Marginal
- Mean confidence: 0.526 (moderate)

This suggests your test environment has significant multipath. For the ML model to learn
distinct LOS vs NLOS correction curves, you'll want to also collect some data in a cleaner
environment (open space, no nearby reflective surfaces within 2m of the modules).

## Files Modified
- `UWB_calibration_ml.py`: +284 lines (1192 -> 1476)
- `uwb_ble_calibration.py`: +161 lines (1273 -> 1434)
- `uwb_dashboard.py`: unchanged
- `dashboard.html`: unchanged
