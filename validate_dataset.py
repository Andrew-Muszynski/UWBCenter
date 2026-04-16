#!/usr/bin/env python3
"""
validate_dataset.py — Post-Collection Data Quality Check
=========================================================
Reads the ble_cal_dataset.csv and prints diagnostics to help decide
whether the data is good enough to train on, or if distances need resampling.

Usage:
    python3 validate_dataset.py [path/to/ble_cal_dataset.csv]

If no path is given, it looks in logs/AD_*/ble_cal_dataset.csv.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd

# ── Thresholds for quality flags ──────────────────────────────────────────────
MIN_SAMPLES_PER_DIST  = 100     # fewer than this = warning
IDEAL_SAMPLES         = 500     # target per distance
MAX_STD_M             = 1.0     # flag if std dev of distance > this
MAX_ERROR_M           = 5.0     # flag if mean error > this
MIN_DISTANCES         = 3       # need at least this many unique distances

# Engineered features (same as uwb_ble_calibration.py)
def engineer(df):
    df = df.copy()
    denom = ((df["fp_ampl2"] + df["fp_ampl3"]) / 2.0).clip(lower=1)
    df["ampl1_ratio"] = df["fp_ampl1"] / denom
    df["cir_norm"] = df["cir_power"] / df["rxpacc"].clip(lower=1)
    df["ampl_spread"] = (df[["fp_ampl2", "fp_ampl3"]].max(axis=1)
                         - df[["fp_ampl2", "fp_ampl3"]].min(axis=1))
    return df


def find_dataset():
    """Auto-find ble_cal_dataset.csv in the logs directory."""
    candidates = sorted(glob.glob("logs/AD_*/ble_cal_dataset.csv"))
    if not candidates:
        candidates = sorted(glob.glob("*/logs/AD_*/ble_cal_dataset.csv"))
    return candidates


def validate(path):
    print(f"\n{'='*62}")
    print(f"  DATASET VALIDATION: {path}")
    print(f"{'='*62}\n")

    if not os.path.exists(path):
        print(f"  ERROR: File not found: {path}")
        return False

    df = pd.read_csv(path)
    n = len(df)

    if n == 0:
        print("  ERROR: Dataset is empty!")
        return False

    print(f"  Total samples     : {n}")
    print(f"  Columns           : {len(df.columns)}")

    # ── Check required columns ────────────────────────────────────────────
    required = {"true_dist_m", "distance_m", "rx_power", "fp_power",
                "fp_rx_ratio", "quality", "std_noise", "fp_ampl1",
                "fp_ampl2", "fp_ampl3", "cir_power", "rxpacc"}
    missing = required - set(df.columns)
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        return False
    print(f"  Required columns  : OK")

    # ── NaN / inf check ───────────────────────────────────────────────────
    num_cols = list(required)
    nan_counts = df[num_cols].isna().sum()
    bad_cols = nan_counts[nan_counts > 0]
    if len(bad_cols) > 0:
        print(f"\n  NaN/missing values:")
        for col, cnt in bad_cols.items():
            pct = cnt / n * 100
            flag = " ⚠" if pct > 5 else ""
            print(f"    {col:<20} {cnt:>6} ({pct:.1f}%){flag}")
    else:
        print(f"  NaN values        : none")

    # ── Distance breakdown ────────────────────────────────────────────────
    dists = sorted(df["true_dist_m"].unique())
    n_dists = len(dists)
    print(f"\n  Unique distances  : {n_dists}  {dists}")

    ok = True
    if n_dists < MIN_DISTANCES:
        print(f"  ⚠  Need at least {MIN_DISTANCES} distances for meaningful regression!")
        ok = False

    print(f"\n  {'True (m)':>10}  {'N':>6}  {'Mean':>8}  {'Std':>8}  {'Error':>8}  {'NLOS%':>6}  Status")
    print(f"  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*12}")

    for d in dists:
        sub = df[df["true_dist_m"] == d]
        cnt = len(sub)
        mean_d = sub["distance_m"].mean()
        std_d = sub["distance_m"].std()
        error = mean_d - d
        nlos_pct = sub["nlos_suspect"].sum() / cnt * 100 if "nlos_suspect" in sub.columns else 0

        flags = []
        if cnt < MIN_SAMPLES_PER_DIST:
            flags.append(f"low-n (<{MIN_SAMPLES_PER_DIST})")
        if std_d > MAX_STD_M:
            flags.append(f"high-std ({std_d:.2f}m)")
        if abs(error) > MAX_ERROR_M:
            flags.append(f"huge-err ({error:+.1f}m)")

        status = ", ".join(flags) if flags else "OK"
        flag_char = "⚠" if flags else "✓"

        print(f"  {d:>10.2f}  {cnt:>6}  {mean_d:>8.3f}  {std_d:>8.3f}  "
              f"{error:>+8.3f}  {nlos_pct:>5.1f}%  {flag_char} {status}")

        if flags:
            ok = False

    # ── Feature statistics ────────────────────────────────────────────────
    df_eng = engineer(df)
    feat_cols = ["distance_m", "rx_power", "fp_power", "fp_rx_ratio", "quality",
                 "ampl1_ratio", "cir_norm", "ampl_spread"]
    print(f"\n  Feature ranges:")
    print(f"  {'Feature':<20}  {'Min':>10}  {'Max':>10}  {'Mean':>10}  {'Std':>10}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for col in feat_cols:
        if col in df_eng.columns:
            vals = df_eng[col].dropna()
            print(f"  {col:<20}  {vals.min():>10.3f}  {vals.max():>10.3f}  "
                  f"{vals.mean():>10.3f}  {vals.std():>10.3f}")

    # ── Correlation with true distance ────────────────────────────────────
    if n_dists >= 2:
        print(f"\n  Feature correlation with true_dist_m:")
        for col in feat_cols:
            if col in df_eng.columns:
                corr = df_eng[col].corr(df_eng["true_dist_m"])
                bar = "█" * max(1, int(abs(corr) * 20))
                print(f"    {col:<20} r={corr:+.3f}  {bar}")

    # ── Antenna delay check ───────────────────────────────────────────────
    if "antenna_delay" in df.columns:
        ad_vals = df["antenna_delay"].unique()
        print(f"\n  Antenna delays    : {sorted(ad_vals.tolist())}")
        if len(ad_vals) > 1:
            print(f"  ⚠  Multiple antenna delays in one dataset — this may confuse the model!")
            ok = False

    # ── Verdict ───────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    if ok:
        print("  ✓ DATASET LOOKS GOOD — ready for training")
    else:
        print("  ⚠ ISSUES FOUND — review warnings above before training")
    print(f"{'='*62}\n")

    return ok


def main():
    if len(sys.argv) > 1:
        paths = [sys.argv[1]]
    else:
        paths = find_dataset()
        if not paths:
            print("\n  No ble_cal_dataset.csv found.")
            print("  Run: python3 validate_dataset.py <path_to_csv>\n")
            sys.exit(1)

    all_ok = True
    for p in paths:
        if not validate(p):
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
