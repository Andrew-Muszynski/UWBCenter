#!/usr/bin/env python3
"""
uwb_ble_calibration.py — BLE-based UWB Calibration & ML Tool
=============================================================
Works alongside uwb_dashboard.py (must be running at localhost:5000).

Workflow:
  1. Collect  — label live BLE packets with known distance + orientation angle
  2. Import   — bulk-load existing logs/tag_*.csv files with labels
  3. Dataset  — view and manage accumulated calibration data
  4. Train    — fit a distance corrector (regression) and optional angle
                classifier (random forest) on the labelled data
  5. Infer    — apply the trained model to the live BLE stream and show a
                corrected position on a 2-D polar map

Requirements:
    pip install bleak scikit-learn numpy pandas matplotlib
    (tkinter is part of the Python standard library)

This file is fully self-contained: collection, training, model save/load, and
inference all live here.  No companion modules or external services required.
"""

import sys
import types
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import threading
import time
import os
import math
import pickle
import asyncio
import struct
import csv

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import deque

try:
    from bleak import BleakScanner, BleakClient
    HAS_BLEAK = True
except ImportError:
    HAS_BLEAK = False

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ═══════════════════════════════ CONFIGURATION ════════════════════════════════

POLL_INTERVAL  = 0.5
DATASET_FILE   = "ble_cal_dataset.csv"
ANGLE_CHOICES  = ["0", "45", "90", "135", "180", "225", "270", "315"]

# ── Direct-BLE configuration (matches anchor_A1_v3.ino / tag_T1_v3.ino) ──────
TAG_CHAR_UUID    = "19b10011-e8f2-537e-4f6c-d104768a1214"
ANCHOR_CHAR_UUID = "19b10012-e8f2-537e-4f6c-d104768a1214"
CMD_CHAR_UUID    = "19b10013-e8f2-537e-4f6c-d104768a1214"

TAG_NAMES    = [f"T{i}" for i in range(1, 11)]
ANCHOR_NAMES = [f"A{i}" for i in range(1, 11)]
ALL_NAMES    = TAG_NAMES + ANCHOR_NAMES

SCAN_TIMEOUT    = 8.0
RESCAN_INTERVAL = 12.0
RECONNECT_SEC   = 3.0
HISTORY_LEN     = 200

# BLE frame layouts (must match the Arduino sketches)
TAG_FMT     = "<BHfiBiBfffHHHHHHBB"   # 43 bytes
ANCHOR_FMT  = "<BHfffHHHHHHiBB"        # 33 bytes
TAG_SIZE    = struct.calcsize(TAG_FMT)
ANCHOR_SIZE = struct.calcsize(ANCHOR_FMT)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════ FEATURES ═════════════════════════════════════

# Raw features available in every tag BLE packet / tag CSV row
RAW_FEATURES = [
    "distance_m",  "rx_power",   "fp_power",  "fp_rx_ratio",
    "quality",     "std_noise",  "fp_ampl1",  "fp_ampl2",
    "fp_ampl3",    "cir_power",  "rxpacc",
]

# Engineered features computed from the raw ones
ENG_FEATURES = [
    "ampl1_ratio",   # fp_ampl1 / mean(fp_ampl2, fp_ampl3)  — 1st-path dominance
    "cir_norm",      # cir_power / rxpacc                    — normalised CIR
    "ampl_spread",   # |fp_ampl2 - fp_ampl3|                 — multipath spread
]

ALL_FEATURES = RAW_FEATURES + ENG_FEATURES

# Sensible defaults for each model
DEFAULT_DIST_FEAT = [
    "distance_m", "fp_rx_ratio", "quality", "ampl1_ratio", "cir_norm", "rx_power",
]
DEFAULT_ANGLE_FEAT = [
    "fp_rx_ratio", "quality", "ampl1_ratio", "ampl_spread", "std_noise", "rx_power",
]

# Columns that must exist in an imported tag CSV
REQUIRED_TAG_COLS = {
    "distance_m", "rx_power", "fp_power", "fp_rx_ratio",
    "quality", "std_noise", "fp_ampl1", "fp_ampl2",
    "fp_ampl3", "cir_power", "rxpacc",
}

# ═══════════════════════════════ DUAL-PIPELINE ENSEMBLE ══════════════════════
# Inlined here so the file is self-contained.  Legacy pickles that reference
# the old `dual_dist_ensemble.DualPipelineEnsemble` import path are remapped
# to this class via the sys.modules shim below.

class DualPipelineEnsemble:
    """Routes / blends two distance pipelines (short-range + long-range).

    `X` is aligned to the union of both feature lists; per-model column
    indices are precomputed at construction time.
    """

    def __init__(
        self,
        model_short,
        feats_short: list,
        model_long,
        feats_long: list,
        union_feats: list,
        raw_dist_idx: int,
        route_threshold_m: float = 0.5,
        blend: str = "route",
        weight_short: float = 0.5,
    ):
        self.model_short = model_short
        self.model_long = model_long
        self.feats_short = list(feats_short)
        self.feats_long = list(feats_long)
        self.union_feats = list(union_feats)
        self.idx_short = [self.union_feats.index(f) for f in self.feats_short]
        self.idx_long = [self.union_feats.index(f) for f in self.feats_long]
        self.raw_dist_idx = int(raw_dist_idx)
        self.route_threshold_m = float(route_threshold_m)
        self.blend = blend
        self.weight_short = float(weight_short)

    def _predict_pair(self, X: np.ndarray):
        Xs = X[:, self.idx_short]
        Xl = X[:, self.idx_long]
        return self.model_short.predict(Xs), self.model_long.predict(Xl)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        ps, pl = self._predict_pair(X)
        if self.blend == "route":
            raw = X[:, self.raw_dist_idx]
            return np.where(raw < self.route_threshold_m, ps, pl)
        if self.blend == "weighted":
            w = self.weight_short
            return w * ps + (1.0 - w) * pl
        raise ValueError(f"Unknown blend mode: {self.blend!r}")

    def __repr__(self) -> str:
        return (
            "DualPipelineEnsemble("
            f"blend={self.blend!r}, threshold={self.route_threshold_m}, "
            f"feats_short={len(self.feats_short)}, feats_long={len(self.feats_long)}, "
            f"union={len(self.union_feats)})"
        )


# Make legacy pickles that reference `dual_dist_ensemble.DualPipelineEnsemble`
# resolve to the inlined class above, so old combined-ensemble .pkl files
# continue to load even though the standalone module no longer exists.
_dde_shim = types.ModuleType("dual_dist_ensemble")
_dde_shim.DualPipelineEnsemble = DualPipelineEnsemble
sys.modules.setdefault("dual_dist_ensemble", _dde_shim)


# ═══════════════════════════════ HELPERS ══════════════════════════════════════

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered feature columns to a DataFrame that has the raw UWB fields."""
    df = df.copy()
    denom = ((df["fp_ampl2"] + df["fp_ampl3"]) / 2.0).clip(lower=1)
    df["ampl1_ratio"] = df["fp_ampl1"] / denom
    df["cir_norm"]    = df["cir_power"] / df["rxpacc"].clip(lower=1)
    df["ampl_spread"] = (df[["fp_ampl2", "fp_ampl3"]].max(axis=1)
                         - df[["fp_ampl2", "fp_ampl3"]].min(axis=1))
    return df


def pkt_to_row(pkt: dict, device: str,
               true_dist: float, angle: float, session_id: str) -> dict:
    """Convert a raw /api/state history-packet dict to a calibration row dict."""
    d = pkt.get("distance_m", float("nan"))
    return {
        "timestamp":    pkt.get("_ts", datetime.now().isoformat(timespec="milliseconds")),
        "device":       device,
        "session_id":   session_id,
        "seq":          pkt.get("seq",          0),
        "true_dist_m":  true_dist,
        "angle_deg":    angle,
        "distance_m":   d,
        "rx_power":     pkt.get("rx_power",     float("nan")),
        "fp_power":     pkt.get("fp_power",     float("nan")),
        "fp_rx_ratio":  pkt.get("fp_rx_ratio",  float("nan")),
        "quality":      pkt.get("quality",      float("nan")),
        "std_noise":    pkt.get("std_noise",    float("nan")),
        "fp_ampl1":     pkt.get("fp_ampl1",     float("nan")),
        "fp_ampl2":     pkt.get("fp_ampl2",     float("nan")),
        "fp_ampl3":     pkt.get("fp_ampl3",     float("nan")),
        "cir_power":    pkt.get("cir_power",    float("nan")),
        "rxpacc":       pkt.get("rxpacc",       float("nan")),
        "nlos_suspect": pkt.get("nlos_suspect", False),
        "anchor_id":    pkt.get("anchor_id",    0),
        "error_m":      d - true_dist if not np.isnan(d) else float("nan"),
    }


def safe_build_X(df: pd.DataFrame, features: list) -> np.ndarray:
    """Extract feature matrix, filling non-finite values with column medians."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    X = df[features].astype(float).values
    X[~np.isfinite(X)] = np.nan
    with np.errstate(all="ignore"):
        col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    bad_idx = np.where(~np.isfinite(X))
    X[bad_idx] = np.take(col_med, bad_idx[1])
    return X


# ═══════════════════════════════ BLE FRAME UNPACKERS ═════════════════════════

def unpack_tag(data: bytes) -> dict | None:
    """Parse a 43-byte TagFrame from the v3 firmware."""
    if len(data) < TAG_SIZE:
        return None
    v = struct.unpack(TAG_FMT, data[:TAG_SIZE])
    round_trip  = (v[4]  << 32) | (v[3]  & 0xFFFFFFFF)
    reply_delay = (v[6]  << 32) | (v[5]  & 0xFFFFFFFF)
    return {
        "anchor_id":    v[0],
        "seq":          v[1],
        "distance_m":   round(v[2], 3),
        "round_trip":   round_trip,
        "reply_delay":  reply_delay,
        "rx_power":     round(v[7], 1),
        "fp_power":     round(v[8], 1),
        "fp_rx_ratio":  round(v[8] - v[7], 1),
        "quality":      round(v[9], 2),
        "std_noise":    v[10],
        "fp_ampl1":     v[11],
        "fp_ampl2":     v[12],
        "fp_ampl3":     v[13],
        "cir_power":    v[14],
        "rxpacc":       v[15],
        "flags":        v[16],
        "anchor_count": v[17],
        "nlos_suspect": bool(v[16] & 0x02),
    }


def unpack_anchor(data: bytes) -> dict | None:
    """Parse a 33-byte AnchorFrame from the v3 firmware."""
    if len(data) < ANCHOR_SIZE:
        return None
    v = struct.unpack(ANCHOR_FMT, data[:ANCHOR_SIZE])
    reply_delay = (v[12] << 32) | (v[11] & 0xFFFFFFFF)
    return {
        "tag_id":       v[0],
        "seq":          v[1],
        "rx_power":     round(v[2], 1),
        "fp_power":     round(v[3], 1),
        "fp_rx_ratio":  round(v[3] - v[2], 1),
        "quality":      round(v[4], 2),
        "std_noise":    v[5],
        "fp_ampl1":     v[6],
        "fp_ampl2":     v[7],
        "fp_ampl3":     v[8],
        "cir_power":    v[9],
        "rxpacc":       v[10],
        "reply_delay":  reply_delay,
        "flags":        v[13],
    }


# ═══════════════════════════════ CSV LOGGING ═════════════════════════════════
# Mirrors uwb_dashboard.py: per-antenna-delay subfolders, separate tag/anchor CSVs.

TAG_HEADER = [
    "timestamp", "device", "anchor_id", "seq", "distance_m",
    "rx_power", "fp_power", "fp_rx_ratio", "quality",
    "round_trip", "reply_delay",
    "std_noise", "fp_ampl1", "fp_ampl2", "fp_ampl3",
    "cir_power", "rxpacc", "flags", "anchor_count", "nlos_suspect",
]
ANCHOR_HEADER = [
    "timestamp", "device", "tag_id", "seq",
    "rx_power", "fp_power", "fp_rx_ratio", "quality",
    "std_noise", "fp_ampl1", "fp_ampl2", "fp_ampl3",
    "cir_power", "rxpacc", "reply_delay", "flags",
]


class StreamLogger:
    """Owns the two open CSV writers for the current antenna-delay subfolder."""

    def __init__(self):
        self._lock = threading.Lock()
        self.antenna_delay = 0
        self.tag_path: Path | None = None
        self.anchor_path: Path | None = None
        self._tag_file = None
        self._anchor_file = None
        self._tag_writer = None
        self._anchor_writer = None
        self.rotate(0)

    def _subdir(self) -> Path:
        ad = self.antenna_delay
        sub = LOG_DIR / (f"AD_{ad}" if ad else "AD_unknown")
        sub.mkdir(parents=True, exist_ok=True)
        return sub

    def rotate(self, antenna_delay: int):
        with self._lock:
            self.antenna_delay = int(antenna_delay)
            self._close_locked()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sub = self._subdir()
            self.tag_path    = sub / f"tag_{ts}.csv"
            self.anchor_path = sub / f"anchor_{ts}.csv"
            self._tag_file    = open(self.tag_path, "w", newline="")
            self._anchor_file = open(self.anchor_path, "w", newline="")
            self._tag_writer    = csv.writer(self._tag_file)
            self._anchor_writer = csv.writer(self._anchor_file)
            self._tag_writer.writerow(TAG_HEADER)
            self._anchor_writer.writerow(ANCHOR_HEADER)

    def _close_locked(self):
        for f in (self._tag_file, self._anchor_file):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        self._tag_file = self._anchor_file = None
        self._tag_writer = self._anchor_writer = None

    def close(self):
        with self._lock:
            self._close_locked()

    def log_tag(self, name: str, pkt: dict):
        with self._lock:
            if self._tag_writer is None:
                return
            self._tag_writer.writerow([
                pkt.get("_ts", datetime.now().isoformat(timespec="milliseconds")),
                name, pkt["anchor_id"], pkt["seq"], pkt["distance_m"],
                pkt["rx_power"], pkt["fp_power"], pkt["fp_rx_ratio"],
                pkt["quality"], pkt["round_trip"], pkt["reply_delay"],
                pkt["std_noise"], pkt["fp_ampl1"], pkt["fp_ampl2"], pkt["fp_ampl3"],
                pkt["cir_power"], pkt["rxpacc"], pkt["flags"],
                pkt["anchor_count"], pkt["nlos_suspect"],
            ])
            self._tag_file.flush()

    def log_anchor(self, name: str, pkt: dict):
        with self._lock:
            if self._anchor_writer is None:
                return
            self._anchor_writer.writerow([
                pkt.get("_ts", datetime.now().isoformat(timespec="milliseconds")),
                name, pkt["tag_id"], pkt["seq"],
                pkt["rx_power"], pkt["fp_power"], pkt["fp_rx_ratio"],
                pkt["quality"], pkt["std_noise"], pkt["fp_ampl1"],
                pkt["fp_ampl2"], pkt["fp_ampl3"], pkt["cir_power"],
                pkt["rxpacc"], pkt["reply_delay"], pkt["flags"],
            ])
            self._anchor_file.flush()


# ═══════════════════════════════ BLE ENGINE ═══════════════════════════════════

class BLEEngine:
    """Self-contained BLE engine.  Runs an asyncio loop on a background thread,
    scans for T*/A* devices, connects to each, subscribes to the v3
    notification characteristic, and dispatches tag packets to subscribed
    callbacks.  Also accepts AD/RI/ST commands for any connected device.

    Public surface (thread-safe; call from the Tk main thread):
        engine.start()
        engine.stop()
        engine.subscribe(cb)            cb(device_name: str, pkt: dict)
        engine.send_command(name, cmd)  cmd is "AD:NNNN" / "RI:NNNN" / "ST"
        engine.send_command_all(cmd)
        engine.snapshot()               -> {name: device_state_dict}
        engine.last_status              -> str
        engine.antenna_delay            -> int (most recently observed AD)
    """

    def __init__(self, logger: StreamLogger):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._stop_evt: asyncio.Event | None = None
        self._scan_task = None
        self._main_task: asyncio.Task | None = None
        self._main_future = None
        self._handlers: dict[str, asyncio.Task] = {}
        self._clients: dict[str, BleakClient] = {}
        self._callbacks: list = []
        self._lock = threading.Lock()
        self._devices: dict[str, dict] = {}
        self._max_seq: dict[str, int] = {}
        self.last_status = "idle"
        self.antenna_delay = 0
        self.logger = logger
        self._running = False

    # ── thread-safe entry points ────────────────────────────────────────────
    def subscribe(self, cb):
        self._callbacks.append(cb)

    def reset_seq(self):
        self._max_seq.clear()

    def start(self):
        if self._running or not HAS_BLEAK:
            return
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop_runner, daemon=True, name="BLEEngineLoop")
        self._loop_thread.start()
        # Schedule the BLE main task on the new loop and keep a handle so
        # shutdown can cancel + await it cleanly.
        fut = asyncio.run_coroutine_threadsafe(self._ble_main(), self._loop)
        self._main_future = fut

    def stop(self):
        if not self._running:
            return
        self._running = False
        loop = self._loop
        if loop and loop.is_running():
            # Run the async shutdown on the BLE loop and block briefly so
            # the in-flight scan/connects unwind before we tear the loop
            # down.  This avoids the "Task was destroyed but it is pending"
            # warning that fires when _ble_main is parked inside
            # BleakScanner.discover() at exit.
            try:
                fut = asyncio.run_coroutine_threadsafe(self._async_stop(), loop)
                fut.result(timeout=10.0)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2.0)

    def snapshot(self) -> dict:
        with self._lock:
            return {n: dict(s) for n, s in self._devices.items()}

    def send_command(self, device: str, cmd: str) -> bool:
        if not (self._loop and self._loop.is_running()):
            return False
        asyncio.run_coroutine_threadsafe(self._send_one(device, cmd), self._loop)
        return True

    def send_command_all(self, cmd: str) -> int:
        names = [n for n, s in self.snapshot().items() if s.get("connected")]
        for n in names:
            self.send_command(n, cmd)
        return len(names)

    # ── internal asyncio plumbing ───────────────────────────────────────────
    def _loop_runner(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _async_stop(self):
        self.last_status = "stopping"
        if self._stop_evt:
            self._stop_evt.set()
        # Cancel per-device handlers and wait for them to settle so they
        # don't get garbage-collected while still pending.
        handlers = [t for t in self._handlers.values() if not t.done()]
        for t in handlers:
            t.cancel()
        if handlers:
            await asyncio.gather(*handlers, return_exceptions=True)
        # Cancel _ble_main if it's still parked inside a discover() call,
        # then await it so the loop can stop without leaking the task.
        main_task = getattr(self, "_main_task", None)
        if main_task is not None and not main_task.done():
            main_task.cancel()
            try:
                await main_task
            except (asyncio.CancelledError, Exception):
                pass
        for c in list(self._clients.values()):
            try:
                await c.disconnect()
            except Exception:
                pass

    async def _ble_main(self):
        self._stop_evt = asyncio.Event()
        self._main_task = asyncio.current_task()
        self.last_status = "scanning"
        while not self._stop_evt.is_set():
            try:
                found = await BleakScanner.discover(timeout=SCAN_TIMEOUT)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.last_status = f"scan error: {e}"
                try:
                    await asyncio.wait_for(self._stop_evt.wait(), timeout=RESCAN_INTERVAL)
                except asyncio.TimeoutError:
                    pass
                continue

            targets = sorted(
                [d for d in found if (d.name or "") in ALL_NAMES],
                key=lambda d: d.name or "")
            if targets:
                self.last_status = "visible: " + ", ".join(d.name for d in targets)
            else:
                self.last_status = "no UWB devices visible"

            for dev in targets:
                name = dev.name
                t = self._handlers.get(name)
                if t is None or t.done():
                    handler = (self._handle_tag if name in TAG_NAMES
                               else self._handle_anchor)
                    self._handlers[name] = asyncio.create_task(handler(dev))

            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=RESCAN_INTERVAL)
            except asyncio.TimeoutError:
                pass

        self.last_status = "stopped"

    def _init_state(self, name: str, addr: str, dev_type: str):
        with self._lock:
            if name not in self._devices:
                self._devices[name] = {
                    "type":         dev_type,
                    "connected":    False,
                    "addr":         addr,
                    "connect_time": "",
                    "packet_count": 0,
                    "settings":     {},
                    "last_seq":     -1,
                }
            self._devices[name]["addr"] = addr

    def _set_connected(self, name: str, addr: str, connected: bool):
        with self._lock:
            d = self._devices.get(name)
            if d is None:
                return
            d["connected"] = connected
            d["addr"] = addr
            if connected:
                d["connect_time"] = datetime.now().isoformat(timespec="seconds")

    def _record_settings(self, name: str, settings_str: str):
        old_ad = self.antenna_delay
        with self._lock:
            d = self._devices.get(name)
            if d is None:
                return
            for part in settings_str.strip().split():
                if part.startswith("AD:"):
                    try:
                        ad_val = int(part[3:])
                        d["settings"]["antenna_delay"] = ad_val
                        self.antenna_delay = ad_val
                    except ValueError:
                        pass
                elif part.startswith("RI:"):
                    try:
                        d["settings"]["range_interval"] = int(part[3:])
                    except ValueError:
                        pass
        if self.antenna_delay != old_ad:
            try:
                self.logger.rotate(self.antenna_delay)
            except Exception as e:
                self.last_status = f"log rotate err: {e}"

    async def _refresh_device(self, name: str, current):
        try:
            fresh = await BleakScanner.find_device_by_name(name, timeout=6.0)
            if fresh:
                return fresh
        except Exception:
            pass
        return current

    def _dispatch_tag_to_callbacks(self, name: str, pkt: dict):
        max_seen = self._max_seq.get(name, -1)
        if pkt["seq"] <= max_seen:
            return
        self._max_seq[name] = pkt["seq"]
        for cb in self._callbacks:
            try:
                cb(name, pkt)
            except Exception:
                pass

    async def _handle_tag(self, ble_device):
        name = ble_device.name
        current = ble_device
        self._init_state(name, current.address, "tag")
        while not (self._stop_evt and self._stop_evt.is_set()):
            try:
                async with BleakClient(current, timeout=15.0) as client:
                    self._clients[name] = client
                    self._set_connected(name, current.address, True)
                    self.last_status = f"connected: {name}"

                    def on_notify(_, data):
                        pkt = unpack_tag(data)
                        if pkt is None:
                            return
                        pkt["_ts"] = datetime.now().isoformat(timespec="milliseconds")
                        with self._lock:
                            d = self._devices.get(name)
                            if d is not None:
                                d["packet_count"] += 1
                                d["last_seq"] = pkt["seq"]
                        try:
                            self.logger.log_tag(name, pkt)
                        except Exception:
                            pass
                        self._dispatch_tag_to_callbacks(name, pkt)

                    await client.start_notify(TAG_CHAR_UUID, on_notify)

                    # Initial settings query so we know the AD value
                    try:
                        await client.write_gatt_char(CMD_CHAR_UUID, b"ST")
                        await asyncio.sleep(0.3)
                        resp = await client.read_gatt_char(CMD_CHAR_UUID)
                        self._record_settings(
                            name, resp.decode("utf-8", errors="replace"))
                    except Exception:
                        pass

                    while client.is_connected and not (self._stop_evt and self._stop_evt.is_set()):
                        await asyncio.sleep(0.5)
            except Exception as e:
                self.last_status = f"{name}: {e}"
            finally:
                self._clients.pop(name, None)
                self._set_connected(name, current.address, False)
            if self._stop_evt and self._stop_evt.is_set():
                break
            await asyncio.sleep(RECONNECT_SEC)
            current = await self._refresh_device(name, current)

    async def _handle_anchor(self, ble_device):
        name = ble_device.name
        current = ble_device
        self._init_state(name, current.address, "anchor")
        while not (self._stop_evt and self._stop_evt.is_set()):
            try:
                async with BleakClient(current, timeout=15.0) as client:
                    self._clients[name] = client
                    self._set_connected(name, current.address, True)
                    self.last_status = f"connected: {name}"

                    def on_notify(_, data):
                        pkt = unpack_anchor(data)
                        if pkt is None:
                            return
                        pkt["_ts"] = datetime.now().isoformat(timespec="milliseconds")
                        with self._lock:
                            d = self._devices.get(name)
                            if d is not None:
                                d["packet_count"] += 1
                                d["last_seq"] = pkt["seq"]
                        try:
                            self.logger.log_anchor(name, pkt)
                        except Exception:
                            pass
                        # Anchor frames don't drive calibration callbacks (no
                        # round_trip) — only tags do.

                    await client.start_notify(ANCHOR_CHAR_UUID, on_notify)

                    try:
                        await client.write_gatt_char(CMD_CHAR_UUID, b"ST")
                        await asyncio.sleep(0.3)
                        resp = await client.read_gatt_char(CMD_CHAR_UUID)
                        self._record_settings(
                            name, resp.decode("utf-8", errors="replace"))
                    except Exception:
                        pass

                    while client.is_connected and not (self._stop_evt and self._stop_evt.is_set()):
                        await asyncio.sleep(0.5)
            except Exception as e:
                self.last_status = f"{name}: {e}"
            finally:
                self._clients.pop(name, None)
                self._set_connected(name, current.address, False)
            if self._stop_evt and self._stop_evt.is_set():
                break
            await asyncio.sleep(RECONNECT_SEC)
            current = await self._refresh_device(name, current)

    async def _send_one(self, device: str, cmd: str):
        client = self._clients.get(device)
        if client is None or not client.is_connected:
            self.last_status = f"send {device} {cmd}: not connected"
            return
        try:
            await client.write_gatt_char(CMD_CHAR_UUID, cmd.encode("utf-8"))
            await asyncio.sleep(0.3)
            resp = await client.read_gatt_char(CMD_CHAR_UUID)
            resp_str = resp.decode("utf-8", errors="replace")
            self._record_settings(device, resp_str)
            self.last_status = f"{device} ← {cmd}  → {resp_str.strip()}"
        except Exception as e:
            self.last_status = f"{device} send err: {e}"


# ═══════════════════════════════ MAIN APP ═════════════════════════════════════

class UWBBLECalApp:

    def __init__(self, root):
        self.root = root
        self.root.title("UWB BLE Calibration & ML Tool")
        self.root.geometry("1350x920")
        self.root.minsize(1100, 800)

        # ── shared data state ──────────────────────────────────────────────
        self.dataset     = pd.DataFrame()
        self.dist_model  = None          # sklearn Pipeline for distance
        self.angle_model = None          # sklearn Pipeline for angle class.
        self.dist_feats  = []
        self.angle_feats = []
        self.model_meta  = {}

        # ── collection state (set when session starts) ─────────────────────
        self._collecting       = False
        self._session_buf      = []
        self._session_id       = ""
        self._session_true_d   = 1.0    # set at session start (thread-safe)
        self._session_angle    = 0.0

        # ── inference state ────────────────────────────────────────────────
        self._infer_active  = False
        self._infer_trail   = deque(maxlen=100)
        self._inf_raw_hist  = deque(maxlen=150)
        self._inf_corr_hist = deque(maxlen=150)

        # ── live collection ring-buffers ───────────────────────────────────
        self._col_raw_buf  = deque(maxlen=120)

        # ── BLE engine (direct bleak, replaces the dashboard sidecar) ──────
        self.logger = StreamLogger()
        self.engine = BLEEngine(self.logger)
        self.engine.subscribe(self._on_packet)
        # Backwards-compatible alias for any older code paths
        self.poller = self.engine
        if HAS_BLEAK:
            self.engine.start()
        else:
            print("[!] bleak not installed — BLE engine disabled.")

        self._build_ui()
        self._load_dataset()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(500, self._tick_connection_panel)

    # ══════════════════════════════ UI CONSTRUCTION ════════════════════════

    def _build_ui(self):
        # Always-visible connection panel above the notebook
        self._build_connection_panel()

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.nb = nb

        self.t_collect = ttk.Frame(nb)
        self.t_import  = ttk.Frame(nb)
        self.t_dataset = ttk.Frame(nb)
        self.t_train   = ttk.Frame(nb)
        self.t_infer   = ttk.Frame(nb)

        nb.add(self.t_collect, text="  Collect  ")
        nb.add(self.t_import,  text="  Import Logs  ")
        nb.add(self.t_dataset, text="  Dataset  ")
        nb.add(self.t_train,   text="  Train  ")
        nb.add(self.t_infer,   text="  Live Inference  ")

        self._build_collect_tab()
        self._build_import_tab()
        self._build_dataset_tab()
        self._build_train_tab()
        self._build_infer_tab()

    def _build_connection_panel(self):
        cf = ttk.LabelFrame(self.root, text="BLE Connection", padding=8)
        cf.pack(fill=tk.X, padx=8, pady=(8, 0))

        # Row 1: scan status + per-device dots
        r1 = ttk.Frame(cf); r1.pack(fill=tk.X)
        self.scan_status_lbl = ttk.Label(
            r1, text="(starting)", foreground="gray", width=42, anchor=tk.W)
        self.scan_status_lbl.pack(side=tk.LEFT)

        self.device_dot_lbls = {}
        for name in ("T1", "A1"):
            tag = ttk.Label(r1, text=f"●  {name}", foreground="gray",
                            font=("TkDefaultFont", 12, "bold"))
            tag.pack(side=tk.LEFT, padx=(16, 4))
            self.device_dot_lbls[name] = tag
        self.device_pkt_lbl = ttk.Label(r1, text="", foreground="gray", anchor=tk.W)
        self.device_pkt_lbl.pack(side=tk.LEFT, padx=(20, 0))

        # Row 2: AD entry + push controls
        r2 = ttk.Frame(cf); r2.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(r2, text="Antenna delay:").pack(side=tk.LEFT)
        self.ad_var = tk.StringVar(value="0")
        ttk.Entry(r2, textvariable=self.ad_var, width=8
                 ).pack(side=tk.LEFT, padx=(4, 6))
        ttk.Label(r2, text="Target:").pack(side=tk.LEFT, padx=(8, 4))
        self.ad_target_var = tk.StringVar(value="ALL")
        ttk.Combobox(r2, textvariable=self.ad_target_var, width=8,
                     state="readonly",
                     values=["ALL", "T1", "A1"]).pack(side=tk.LEFT)
        ttk.Button(r2, text="Push", command=self._push_antenna_delay
                  ).pack(side=tk.LEFT, padx=(8, 4))
        ttk.Button(r2, text="Send AD:0 (raw mode)",
                   command=lambda: self._push_ad_value(0, target="ALL")
                  ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(r2, text="Query (ST)", command=self._push_st
                  ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(r2, text="Diagnose…", command=self._diagnose_connection
                  ).pack(side=tk.LEFT, padx=(8, 0))
        self.engine_status_lbl = ttk.Label(r2, text="", foreground="gray")
        self.engine_status_lbl.pack(side=tk.LEFT, padx=(16, 0))

    def _tick_connection_panel(self):
        try:
            snap = self.engine.snapshot()
            ad = self.engine.antenna_delay
            self.scan_status_lbl.configure(
                text=f"{self.engine.last_status}    AD={ad}",
                foreground=("green" if any(d["connected"] for d in snap.values())
                            else "gray"))
            packet_lines = []
            for name, lbl in self.device_dot_lbls.items():
                d = snap.get(name)
                if d is None:
                    lbl.configure(foreground="gray")
                    continue
                lbl.configure(foreground=("#22aa55" if d["connected"] else "#cc4444"))
                packet_lines.append(f"{name}: {d['packet_count']} pkts")
            self.device_pkt_lbl.configure(text="    ".join(packet_lines))
            self.engine_status_lbl.configure(
                text=self.engine.last_status[:80] if self.engine.last_status else "")
        finally:
            self.root.after(500, self._tick_connection_panel)

    def _push_antenna_delay(self):
        try:
            ad = int(self.ad_var.get())
        except ValueError as e:
            messagebox.showerror("Bad AD", str(e))
            return
        if not 0 <= ad <= 65535:
            messagebox.showerror("Bad AD", "AD must be 0..65535")
            return
        self._push_ad_value(ad, target=self.ad_target_var.get())

    def _push_ad_value(self, ad: int, target: str = "ALL"):
        cmd = f"AD:{int(ad)}"
        if target == "ALL":
            n = self.engine.send_command_all(cmd)
            self.engine_status_lbl.configure(
                text=f"queued {cmd} → {n} device(s)", foreground="#22aa55")
        else:
            ok = self.engine.send_command(target, cmd)
            self.engine_status_lbl.configure(
                text=f"queued {cmd} → {target}" if ok else f"queue failed",
                foreground=("#22aa55" if ok else "#cc4444"))
        self.ad_var.set(str(int(ad)))

    def _push_st(self):
        n = self.engine.send_command_all("ST")
        self.engine_status_lbl.configure(
            text=f"queued ST → {n} device(s)", foreground="gray")

    def _diagnose_connection(self):
        """Pop up a one-shot snapshot of BLE engine + adapter state."""
        snap = self.engine.snapshot()
        ad = self.engine.antenna_delay
        lines = [
            f"BLE engine status : {self.engine.last_status or '(idle)'}",
            f"Current antenna δ : AD={ad}",
            f"Devices known     : {len(snap)}",
            "",
        ]
        if snap:
            lines.append(f"  {'name':<6} {'type':<7} {'state':<10} {'packets':>8}  {'last_seq':>9}  addr")
            for name, d in sorted(snap.items()):
                state = "CONNECTED" if d["connected"] else "disconnected"
                lines.append(
                    f"  {name:<6} {d['type']:<7} {state:<10} "
                    f"{d['packet_count']:>8}  {d['last_seq']:>9}  {d['addr'] or '?'}")
        else:
            lines.append("  (no devices discovered yet — scan still in progress)")

        lines.append("")
        lines.append("BLE adapter is owned exclusively by this app — make sure no")
        lines.append("other process (e.g. an old dashboard sidecar) is bound to it.")

        win = tk.Toplevel(self.root)
        win.title("BLE Diagnostics")
        win.geometry("640x340")
        txt = tk.Text(win, font=("Menlo", 10), wrap=tk.NONE,
                      bg="#1a1a2e", fg="#c8d0e0", insertbackground="#c8d0e0")
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        txt.insert("1.0", "\n".join(lines))
        txt.config(state=tk.DISABLED)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 8))

    def _on_close(self):
        try:
            self.engine.stop()
        except Exception:
            pass
        try:
            self.logger.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    # ── Tab 1: Collect ─────────────────────────────────────────────────────

    def _build_collect_tab(self):
        t = self.t_collect

        # Session config
        cf = ttk.LabelFrame(t, text="Session Configuration", padding=8)
        cf.pack(fill=tk.X, padx=10, pady=4)
        r1 = ttk.Frame(cf); r1.pack(fill=tk.X, pady=2)

        ttk.Label(r1, text="True Distance (m):").pack(side=tk.LEFT, padx=(0, 4))
        self.col_dist_var = tk.StringVar(value="1.00")
        ttk.Entry(r1, textvariable=self.col_dist_var, width=8).pack(side=tk.LEFT, padx=(0, 18))

        ttk.Label(r1, text="Orientation (°):").pack(side=tk.LEFT, padx=(0, 4))
        self.col_angle_var = tk.StringVar(value="0")
        ttk.Combobox(r1, textvariable=self.col_angle_var, values=ANGLE_CHOICES,
                     width=6, state="readonly").pack(side=tk.LEFT, padx=(0, 18))

        ttk.Label(r1, text="Notes:").pack(side=tk.LEFT, padx=(0, 4))
        self.col_notes_var = tk.StringVar()
        ttk.Entry(r1, textvariable=self.col_notes_var,
                  width=22).pack(side=tk.LEFT)

        # Angle hint
        hint = ("Orientation convention:  0° = tag antenna facing anchor  "
                "90° = broadside  180° = rear  (mark what you physically set up)")
        ttk.Label(cf, text=hint, foreground="gray",
                  font=("TkDefaultFont", 8)).pack(anchor=tk.W, pady=(2, 0))

        # Controls
        ctl = ttk.LabelFrame(t, text="Controls", padding=8)
        ctl.pack(fill=tk.X, padx=10, pady=4)
        cr = ttk.Frame(ctl); cr.pack(fill=tk.X)

        self.start_btn = ttk.Button(cr, text="▶  Start Session",
                                    command=self._start_session)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_btn = ttk.Button(cr, text="■  Stop & Save",
                                   command=self._stop_session, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 20))

        self.col_count_lbl = ttk.Label(cr, text="Samples: 0",
                                       font=("Helvetica", 12, "bold"))
        self.col_count_lbl.pack(side=tk.LEFT, padx=(0, 20))
        self.col_stats_lbl = ttk.Label(cr, text="Mean: —   Std: —   Error: —")
        self.col_stats_lbl.pack(side=tk.LEFT)

        # Live mini-plot
        pf = ttk.LabelFrame(t, text="Live Distance — last 120 samples", padding=5)
        pf.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 10))
        self._col_fig = Figure(figsize=(9, 3), dpi=96)
        self._col_ax  = self._col_fig.add_subplot(111)
        self._col_canvas = FigureCanvasTkAgg(self._col_fig, master=pf)
        self._col_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Tab 2: Import Logs ──────────────────────────────────────────────────

    def _build_import_tab(self):
        t = self.t_import

        info = ttk.LabelFrame(t, text="Import Existing tag_*.csv Logs", padding=10)
        info.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(info, wraplength=900,
            text=(
                "Select one or more tag CSV files from logs/.  "
                "If a file already has a populated 'true_dist_m' column (multi-distance session), "
                "those per-row labels are preserved and no prompt is shown.  "
                "Otherwise the defaults below are used — or, if 'Ask per file' is checked, "
                "you will be prompted for each file individually."
            )).pack(anchor=tk.W)

        dr = ttk.Frame(info); dr.pack(fill=tk.X, pady=6)
        ttk.Label(dr, text="Default true distance (m):").pack(side=tk.LEFT, padx=(0, 4))
        self.imp_dist_var = tk.StringVar(value="1.00")
        ttk.Entry(dr, textvariable=self.imp_dist_var, width=8).pack(side=tk.LEFT, padx=(0, 18))

        ttk.Label(dr, text="Default angle (°):").pack(side=tk.LEFT, padx=(0, 4))
        self.imp_angle_var = tk.StringVar(value="0")
        ttk.Combobox(dr, textvariable=self.imp_angle_var, values=ANGLE_CHOICES,
                     width=6, state="readonly").pack(side=tk.LEFT, padx=(0, 18))

        self.imp_ask_each = tk.BooleanVar(value=True)
        ttk.Checkbutton(dr, text="Ask for distance/angle per file",
                        variable=self.imp_ask_each).pack(side=tk.LEFT)

        br = ttk.Frame(info); br.pack(fill=tk.X, pady=4)
        ttk.Button(br, text="Browse & Import…",
                   command=self._import_logs).pack(side=tk.LEFT, padx=(0, 10))
        self.imp_status_lbl = ttk.Label(br, text="")
        self.imp_status_lbl.pack(side=tk.LEFT)

        lf = ttk.LabelFrame(t, text="Import Log", padding=5)
        lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.imp_log_text = tk.Text(lf, height=24, font=("Consolas", 9),
                                    bg="#1a1a2e", fg="#c8d0e0", state=tk.DISABLED)
        sb = ttk.Scrollbar(lf, command=self.imp_log_text.yview)
        self.imp_log_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.imp_log_text.pack(fill=tk.BOTH, expand=True)

    # ── Tab 3: Dataset ──────────────────────────────────────────────────────

    def _build_dataset_tab(self):
        t = self.t_dataset

        ctl = ttk.Frame(t); ctl.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(ctl, text="Refresh",
                   command=self._refresh_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctl, text="Export CSV",
                   command=self._export_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctl, text="Clear All",
                   command=self._clear_dataset).pack(side=tk.LEFT, padx=2)
        self.ds_lbl = ttk.Label(ctl, text="0 samples")
        self.ds_lbl.pack(side=tk.RIGHT, padx=10)

        cols = ("true_dist", "angle", "distance_m", "error",
                "rx_power", "fp_rx_ratio", "quality",
                "ampl1_ratio", "nlos", "device")
        self.ds_tree = ttk.Treeview(t, columns=cols, show="headings", height=18)
        for c, h, w in [
            ("true_dist",  "True (m)",   80), ("angle",     "Angle °",  60),
            ("distance_m", "Meas (m)",   80), ("error",     "Err (m)",  75),
            ("rx_power",   "RxPwr",      65), ("fp_rx_ratio","FP-RX",   60),
            ("quality",    "Quality",    65), ("ampl1_ratio","Ampl1R",  70),
            ("nlos",       "NLOS?",      50), ("device",    "Device",   55),
        ]:
            self.ds_tree.heading(c, text=h)
            self.ds_tree.column(c, width=w, anchor=tk.CENTER)

        vsc = ttk.Scrollbar(t, orient=tk.VERTICAL, command=self.ds_tree.yview)
        self.ds_tree.configure(yscrollcommand=vsc.set)
        vsc.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5))
        self.ds_tree.pack(fill=tk.BOTH, expand=True, padx=(10, 0))

        sf = ttk.LabelFrame(t, text="Summary by Distance × Angle", padding=6)
        sf.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.ds_sum = tk.Text(sf, height=8, font=("Consolas", 9),
                              bg="#1a1a2e", fg="#c8d0e0",
                              insertbackground="#c8d0e0", state=tk.DISABLED)
        self.ds_sum.pack(fill=tk.X)

    # ── Tab 4: Train ────────────────────────────────────────────────────────

    def _build_train_tab(self):
        t = self.t_train
        top = ttk.Frame(t); top.pack(fill=tk.BOTH, expand=True)

        # ── Left panel: config ──
        lf = ttk.Frame(top, width=400)
        lf.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        lf.pack_propagate(False)

        # Distance corrector
        dc = ttk.LabelFrame(lf, text="Distance Corrector", padding=8)
        dc.pack(fill=tk.X, pady=(0, 6))
        ra = ttk.Frame(dc); ra.pack(fill=tk.X)
        ttk.Label(ra, text="Algorithm:").pack(side=tk.LEFT)
        self.dist_algo_var = tk.StringVar(value="gradient_boosting")
        ttk.Combobox(ra, textvariable=self.dist_algo_var, width=20, state="readonly",
                     values=["gradient_boosting", "random_forest", "ridge_poly3"]
                     ).pack(side=tk.LEFT, padx=4)

        ttk.Label(dc, text="Input features:", anchor=tk.W).pack(anchor=tk.W, pady=(6, 0))
        self.dist_feat_vars = {}
        fg = ttk.Frame(dc); fg.pack(fill=tk.X)
        for i, f in enumerate(ALL_FEATURES):
            v = tk.BooleanVar(value=(f in DEFAULT_DIST_FEAT))
            self.dist_feat_vars[f] = v
            ttk.Checkbutton(fg, text=f, variable=v
                            ).grid(row=i // 2, column=i % 2, sticky=tk.W, padx=4, pady=1)

        # Angle classifier
        ac = ttk.LabelFrame(lf, text="Angle Classifier  (optional)", padding=8)
        ac.pack(fill=tk.X, pady=(0, 6))
        self.train_angle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ac, text="Train angle classifier (needs ≥ 2 distinct angles in dataset)",
                        variable=self.train_angle_var).pack(anchor=tk.W)
        ttk.Label(ac, text="Input features:", anchor=tk.W).pack(anchor=tk.W, pady=(4, 0))
        self.angle_feat_vars = {}
        afg = ttk.Frame(ac); afg.pack(fill=tk.X)
        for i, f in enumerate(ALL_FEATURES):
            v = tk.BooleanVar(value=(f in DEFAULT_ANGLE_FEAT))
            self.angle_feat_vars[f] = v
            ttk.Checkbutton(afg, text=f, variable=v
                            ).grid(row=i // 2, column=i % 2, sticky=tk.W, padx=4, pady=1)

        # Buttons
        brow = ttk.Frame(lf); brow.pack(fill=tk.X, pady=4)
        ttk.Button(brow, text="⚙  Train",
                   command=self._train_models).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(brow, text="Save…",
                   command=self._save_models).pack(side=tk.LEFT, padx=2)
        ttk.Button(brow, text="Load…",
                   command=self._load_models).pack(side=tk.LEFT, padx=2)
        self.train_status = ttk.Label(brow, text="No model", foreground="gray")
        self.train_status.pack(side=tk.LEFT, padx=10)

        # ── Right panel: results ──
        rf = ttk.Frame(top)
        rf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.train_results = tk.Text(rf, height=14, font=("Consolas", 9),
                                     bg="#1a1a2e", fg="#c8d0e0",
                                     insertbackground="#c8d0e0",
                                     state=tk.DISABLED, wrap=tk.WORD)
        self.train_results.pack(fill=tk.X, pady=(0, 4))
        self._tr_fig    = Figure(figsize=(7, 4), dpi=96)
        self._tr_canvas = FigureCanvasTkAgg(self._tr_fig, master=rf)
        self._tr_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Tab 5: Live Inference ───────────────────────────────────────────────

    def _build_infer_tab(self):
        t = self.t_infer

        ctrl = ttk.Frame(t); ctrl.pack(fill=tk.X, padx=10, pady=8)
        self.infer_btn = ttk.Button(ctrl, text="▶  Start Inference (live BLE)",
                                    command=self._toggle_inference)
        self.infer_btn.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Button(ctrl, text="▶  Replay loaded dataset",
                   command=self._replay_dataset
                   ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(ctrl, text="▶  Replay CSV file…",
                   command=self._replay_dataset_file
                   ).pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(ctrl, text="True dist (m):").pack(side=tk.LEFT)
        self.infer_true_var = tk.StringVar(value="1.00")
        ttk.Entry(ctrl, textvariable=self.infer_true_var,
                  width=7).pack(side=tk.LEFT, padx=(2, 14))

        ttk.Label(ctrl, text="True angle (°):").pack(side=tk.LEFT)
        self.infer_true_angle_var = tk.StringVar(value="0")
        ttk.Combobox(ctrl, textvariable=self.infer_true_angle_var,
                     values=ANGLE_CHOICES, width=5, state="readonly"
                     ).pack(side=tk.LEFT, padx=(2, 0))

        self.infer_status = ttk.Label(ctrl, text="Inactive", foreground="gray")
        self.infer_status.pack(side=tk.LEFT, padx=14)
        self._infer_pkt_seen = 0
        self._infer_started_at = 0.0

        # Metric panels
        mf = ttk.Frame(t); mf.pack(fill=tk.X, padx=10, pady=(0, 5))
        panels = [
            ("i_raw_lbl",  "Raw Distance",  "#ffffff"),
            ("i_corr_lbl", "Corrected Dist","#00cc88"),
            ("i_err_lbl",  "Corr. Error",   "#ffaa00"),
            ("i_angle_lbl","Est. Angle",    "#88aaff"),
        ]
        for attr, label, color in panels:
            pnl = ttk.LabelFrame(mf, text=label, padding=8)
            pnl.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
            lbl = ttk.Label(pnl, text="—", font=("Helvetica", 18, "bold"),
                            foreground=color)
            lbl.pack()
            setattr(self, attr, lbl)

        # Position map + time series
        bot = ttk.Frame(t)
        bot.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        self._inf_fig     = Figure(figsize=(12, 4), dpi=96)
        self._inf_fig.patch.set_facecolor("#0d0d0d")
        self._inf_ax_pos  = self._inf_fig.add_subplot(121)
        self._inf_ax_dist = self._inf_fig.add_subplot(122)
        self._inf_canvas  = FigureCanvasTkAgg(self._inf_fig, master=bot)
        self._inf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ══════════════════════ COLLECTION ════════════════════════════════════

    def _start_session(self):
        try:
            td  = float(self.col_dist_var.get())
            ang = float(self.col_angle_var.get())
        except ValueError:
            messagebox.showwarning("Invalid", "Enter a valid true distance.")
            return

        self._session_true_d = td
        self._session_angle  = ang
        self._session_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_buf    = []
        self._col_raw_buf.clear()
        self._collecting     = True

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.col_count_lbl.config(text="Samples: 0")
        self.col_stats_lbl.config(text="Collecting…")

    def _stop_session(self):
        self._collecting = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        n = len(self._session_buf)
        if n == 0:
            messagebox.showinfo("Empty Session", "No packets were collected.")
            return

        new_df = pd.DataFrame(self._session_buf)
        new_df = engineer(new_df)
        self.dataset = pd.concat([self.dataset, new_df], ignore_index=True)
        self._save_dataset()

        d = new_df["distance_m"]
        td = new_df["true_dist_m"].iloc[0]
        ang = new_df["angle_deg"].iloc[0]
        messagebox.showinfo("Session Saved",
            f"{n} samples saved.\n\n"
            f"True dist : {td:.3f} m    Angle : {ang:.0f}°\n"
            f"Mean meas : {d.mean():.3f} m\n"
            f"Error     : {d.mean() - td:+.3f} m\n"
            f"Std       : {d.std():.3f} m\n\n"
            f"Dataset total: {len(self.dataset)} samples")
        self._refresh_dataset()

    def _on_packet(self, tag_name: str, pkt: dict):
        """Called from the BLE poller background thread for every new packet."""
        if self._collecting:
            row = pkt_to_row(pkt, tag_name,
                             self._session_true_d,
                             self._session_angle,
                             self._session_id)
            # Guard dataset quality against BLE parse glitches that can produce +/-inf.
            for c in RAW_FEATURES + ["true_dist_m", "angle_deg", "error_m"]:
                v = row.get(c, float("nan"))
                if isinstance(v, (int, float, np.number)) and not np.isfinite(v):
                    row[c] = float("nan")
            self._session_buf.append(row)
            d = pkt.get("distance_m", float("nan"))
            if not np.isnan(d):
                self._col_raw_buf.append(d)
            # Schedule UI update on main thread
            self.root.after(0, self._update_collect_ui)

        if self._infer_active:
            self.root.after(0, self._update_inference, pkt)

    def _update_collect_ui(self):
        n   = len(self._session_buf)
        arr = np.array([v for v in self._col_raw_buf if not np.isnan(v)])
        self.col_count_lbl.config(text=f"Samples: {n}")
        if len(arr) > 0:
            td = self._session_true_d
            self.col_stats_lbl.config(
                text=(f"Mean: {arr.mean():.3f} m   "
                      f"Std: {arr.std():.3f} m   "
                      f"Error: {arr.mean() - td:+.3f} m"))
        if n % 5 == 0 or n < 5:
            self._redraw_col_plot()

    def _redraw_col_plot(self):
        ax = self._col_ax
        ax.clear()
        data = list(self._col_raw_buf)
        if data:
            ax.plot(data, color="#4488ff", alpha=0.7, lw=0.9, label="Raw dist")
            ax.axhline(self._session_true_d, color="orange", ls="--", lw=1.2,
                       label=f"True {self._session_true_d:.2f} m")
            lo = max(0, self._session_true_d - 2)
            hi = self._session_true_d + 2
            ax.set_ylim(lo, hi)
            ax.set_ylabel("Distance (m)")
            ax.set_xlabel("Sample")
            ax.legend(fontsize=8, loc="upper right")
        self._col_fig.tight_layout()
        self._col_canvas.draw_idle()

    # ══════════════════════ IMPORT LOGS ════════════════════════════════════

    def _import_logs(self):
        log_dir = Path("logs") if Path("logs").exists() else Path(".")
        paths = filedialog.askopenfilenames(
            title="Select tag_*.csv files",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            initialdir=str(log_dir),
            parent=self.root)
        if not paths:
            return

        total = 0
        for p in paths:
            try:
                total += self._import_one(p)
            except Exception as e:
                self._imp_log(f"ERROR  {os.path.basename(p)}: {e}\n")

        if total > 0:
            self._save_dataset()
            self._refresh_dataset()
            self.imp_status_lbl.config(
                text=f"Imported {total} samples.  Dataset total: {len(self.dataset)}")

    def _import_one(self, path: str) -> int:
        name = os.path.basename(path)
        df   = pd.read_csv(path)
        df = df.replace([np.inf, -np.inf], np.nan)

        if not REQUIRED_TAG_COLS.issubset(df.columns):
            missing = REQUIRED_TAG_COLS - set(df.columns)
            self._imp_log(f"SKIP  {name} — missing columns: {missing}\n")
            return 0

        # Multi-distance file: file already has per-row true_dist_m → preserve as-is
        if "true_dist_m" in df.columns:
            td_series = pd.to_numeric(df["true_dist_m"], errors="coerce")
            if td_series.notna().any() and (td_series.fillna(0) != 0).any():
                df["true_dist_m"] = td_series
                df = df[df["true_dist_m"].notna()].reset_index(drop=True)

                if "angle_deg" in df.columns and pd.to_numeric(
                        df["angle_deg"], errors="coerce").notna().any():
                    df["angle_deg"] = pd.to_numeric(df["angle_deg"], errors="coerce").fillna(0)
                else:
                    df["angle_deg"] = float(self.imp_angle_var.get() or 0)

                df["error_m"]    = df["distance_m"] - df["true_dist_m"]
                df["session_id"] = name
                if "nlos_suspect" not in df.columns:
                    df["nlos_suspect"] = False

                df = engineer(df)
                self.dataset = pd.concat([self.dataset, df], ignore_index=True)
                summary = ", ".join(
                    f"{d:g}m×{c}" for d, c in
                    sorted(df["true_dist_m"].value_counts().items()))
                self._imp_log(
                    f"OK    {name}: {len(df)} samples  (multi-distance: {summary})\n")
                return len(df)

        # Single-label file: prompt or use defaults (existing behaviour)
        if self.imp_ask_each.get():
            td_str = simpledialog.askstring(
                "True Distance",
                f"True distance (m) for:\n{name}",
                initialvalue=self.imp_dist_var.get(),
                parent=self.root)
            if td_str is None:
                self._imp_log(f"SKIP  {name} — cancelled\n")
                return 0
            ang_str = simpledialog.askstring(
                "Orientation Angle",
                f"Tag orientation angle (°) for:\n{name}",
                initialvalue=self.imp_angle_var.get(),
                parent=self.root)
            ang_str = ang_str or "0"
        else:
            td_str  = self.imp_dist_var.get()
            ang_str = self.imp_angle_var.get()

        td  = float(td_str)
        ang = float(ang_str)

        df["true_dist_m"]  = td
        df["angle_deg"]    = ang
        df["error_m"]      = df["distance_m"] - td
        df["session_id"]   = name

        if "nlos_suspect" not in df.columns:
            df["nlos_suspect"] = False

        df = engineer(df)
        self.dataset = pd.concat([self.dataset, df], ignore_index=True)
        self._imp_log(
            f"OK    {name}: {len(df)} samples  @  {td:.3f} m / {ang:.0f}°\n")
        return len(df)

    def _imp_log(self, text: str):
        self.imp_log_text.config(state=tk.NORMAL)
        self.imp_log_text.insert(tk.END, text)
        self.imp_log_text.see(tk.END)
        self.imp_log_text.config(state=tk.DISABLED)

    # ══════════════════════ DATASET MANAGEMENT ═════════════════════════════

    def _save_dataset(self):
        if not self.dataset.empty:
            self.dataset.to_csv(DATASET_FILE, index=False)

    def _load_dataset(self):
        if os.path.exists(DATASET_FILE):
            try:
                df = pd.read_csv(DATASET_FILE)
                df = df.replace([np.inf, -np.inf], np.nan)
                self.dataset = engineer(df)
            except Exception:
                self.dataset = pd.DataFrame()

    def _refresh_dataset(self):
        for item in self.ds_tree.get_children():
            self.ds_tree.delete(item)

        if self.dataset.empty:
            self.ds_lbl.config(text="0 samples")
            return

        df   = self.dataset
        show = df.tail(600)
        for _, r in show.iterrows():
            ar = r.get("ampl1_ratio", float("nan"))
            self.ds_tree.insert("", tk.END, values=(
                f"{r.get('true_dist_m', 0):.3f}",
                f"{r.get('angle_deg', 0):.0f}",
                f"{r.get('distance_m', 0):.3f}",
                f"{r.get('error_m', 0):+.3f}",
                f"{r.get('rx_power', 0):.1f}",
                f"{r.get('fp_rx_ratio', 0):.1f}",
                f"{r.get('quality', 0):.1f}",
                f"{ar:.2f}" if not np.isnan(ar) else "—",
                str(bool(r.get("nlos_suspect", False))),
                str(r.get("device", "?")),
            ))

        self.ds_lbl.config(text=f"{len(df)} samples  (showing {len(show)})")
        self._refresh_summary(df)

    def _refresh_summary(self, df: pd.DataFrame):
        if "true_dist_m" not in df.columns:
            return

        gcols = ["true_dist_m"]
        if "angle_deg" in df.columns and df["angle_deg"].nunique() > 1:
            gcols.append("angle_deg")

        hdr = (f"{'Configuration':<32} {'N':>5}  {'Mean Meas':>10}  "
               f"{'Mean Err':>10}  {'Std':>8}  {'MAE':>8}")
        lines = [hdr, "─" * 80]

        # Pass a plain string (not a list) when there's only one group key so
        # pandas returns scalar names instead of 1-tuples, avoiding IndexError.
        group_key = gcols[0] if len(gcols) == 1 else gcols
        for name, grp in df.groupby(group_key):
            if isinstance(name, tuple):
                lbl = f"{name[0]:.2f} m @ {name[1]:.0f}°"
            else:
                lbl = f"{name:.2f} m"
            err = grp["distance_m"] - grp["true_dist_m"]
            lines.append(
                f"  {lbl:<30}{len(grp):>5}  {grp['distance_m'].mean():>10.3f}  "
                f"{err.mean():>+10.3f}  {err.std():>8.3f}  {err.abs().mean():>8.3f}")

        lines.append("─" * 80)
        te = df["distance_m"] - df["true_dist_m"]
        lines.append(
            f"  {'TOTAL':<30}{len(df):>5}  {df['distance_m'].mean():>10.3f}  "
            f"{te.mean():>+10.3f}  {te.std():>8.3f}  {te.abs().mean():>8.3f}")

        self.ds_sum.config(state=tk.NORMAL)
        self.ds_sum.delete("1.0", tk.END)
        self.ds_sum.insert("1.0", "\n".join(lines))
        self.ds_sum.config(state=tk.DISABLED)

    def _export_dataset(self):
        if self.dataset.empty:
            messagebox.showinfo("Empty", "No data to export.")
            return
        LOG_DIR.mkdir(exist_ok=True)
        default_name = f"dataset_export_{datetime.now():%Y%m%d_%H%M%S}.csv"
        p = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialdir=str(LOG_DIR.resolve()),
            initialfile=default_name,
            parent=self.root)
        if not p:
            return
        try:
            self.dataset.to_csv(p, index=False)
        except Exception as e:
            messagebox.showerror("Export failed",
                f"Could not write CSV:\n{p}\n\n{type(e).__name__}: {e}",
                parent=self.root)
            return
        messagebox.showinfo("Saved",
            f"Exported {len(self.dataset)} rows to:\n{p}",
            parent=self.root)

    def _clear_dataset(self):
        if messagebox.askyesno("Clear", "Delete ALL calibration data?"):
            self.dataset = pd.DataFrame()
            if os.path.exists(DATASET_FILE):
                os.remove(DATASET_FILE)
            self._refresh_dataset()

    # ══════════════════════ ML TRAINING ════════════════════════════════════

    def _get_feat_list(self, var_dict: dict) -> list:
        return [k for k, v in var_dict.items() if v.get()]

    def _make_dist_pipeline(self, algo: str) -> Pipeline:
        if algo == "gradient_boosting":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("reg", GradientBoostingRegressor(
                    n_estimators=400, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42))])
        elif algo == "random_forest":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("reg", RandomForestRegressor(
                    n_estimators=400, max_depth=10, random_state=42))])
        else:  # ridge_poly3
            return Pipeline([
                ("poly", PolynomialFeatures(degree=3, include_bias=False)),
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0))])

    def _train_models(self):
        if self.dataset.empty or len(self.dataset) < 20:
            messagebox.showwarning(
                "Insufficient Data",
                f"Need ≥ 20 labelled samples.  Have {len(self.dataset)}.")
            return

        raw_rows = len(self.dataset)
        df = engineer(
            self.dataset.dropna(subset=["distance_m", "true_dist_m"]).copy()
        )
        dropped = raw_rows - len(df)
        dist_feats  = self._get_feat_list(self.dist_feat_vars)
        angle_feats = self._get_feat_list(self.angle_feat_vars)

        if not dist_feats:
            messagebox.showwarning("No Features",
                                   "Select at least one distance feature.")
            return

        algo = self.dist_algo_var.get()
        ALGO_BLURB = {
            "gradient_boosting":
                "GradientBoostingRegressor — additive trees, good on small/mid "
                "tabular sets, captures non-linear bias well",
            "random_forest":
                "RandomForestRegressor — bagged trees, robust to noise, slower "
                "but less hyper-param-sensitive",
            "ridge_poly3":
                "Ridge regression on degree-3 polynomial features — smooth, "
                "interpretable, good when calibration is mostly polynomial",
        }

        # Per-distance count of usable rows (what the model will actually see)
        dist_counts = (df["true_dist_m"]
                       .round(3).value_counts().sort_index().to_dict())
        per_dist_str = "  ".join(f"{d}m×{n}" for d, n in dist_counts.items())

        txt = [
            f"{'='*64}",
            "  ML TRAINING REPORT",
            f"{'='*64}\n",
            "INPUTS",
            f"  Dataset file       : {DATASET_FILE}",
            f"  Total rows in CSV  : {raw_rows}",
            f"  Usable rows        : {len(df)}   "
                f"({dropped} dropped: missing distance_m or true_dist_m)",
            f"  Distinct distances : {df['true_dist_m'].nunique()}    {per_dist_str}",
            f"  Distinct angles    : {df['angle_deg'].nunique() if 'angle_deg' in df.columns else 0}",
            "",
            "ALGORITHM",
            f"  Choice  : {algo}",
            f"  About   : {ALGO_BLURB.get(algo, '')}",
            f"  Features ({len(dist_feats)}): {', '.join(dist_feats)}",
            f"  Target   : true_dist_m  (m)",
            f"  Split    : 80 % train / 20 % test  (random_state=42)",
            "",
        ]

        # ── Distance corrector ─────────────────────────────────────────────
        try:
            X = safe_build_X(df, dist_feats)
        except ValueError as e:
            messagebox.showerror("Feature Error", str(e))
            return

        y = df["true_dist_m"].values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42)

        pipeline = self._make_dist_pipeline(algo)
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)

        di       = dist_feats.index("distance_m") if "distance_m" in dist_feats else None
        raw_te   = X_te[:, di] if di is not None else y_te
        raw_mae  = float(np.mean(np.abs(raw_te - y_te)))
        cor_mae  = float(mean_absolute_error(y_te, y_pred))
        cor_rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2       = float(r2_score(y_te, y_pred))
        improv   = (raw_mae - cor_mae) / raw_mae * 100 if raw_mae > 0 else 0

        cv_k = min(5, max(2, len(X) // 10))
        cv   = cross_val_score(pipeline, X, y, cv=cv_k,
                               scoring="neg_mean_absolute_error")

        txt += [
            f"DISTANCE CORRECTOR  ({algo})",
            f"  Features : {', '.join(dist_feats)}",
            f"  Samples  : {len(X_tr)} train / {len(X_te)} test\n",
            f"  Raw  MAE : {raw_mae*100:6.2f} cm",
            f"  Corr MAE : {cor_mae*100:6.2f} cm   (↑ {improv:.1f}% improvement)",
            f"  Corr RMSE: {cor_rmse*100:6.2f} cm",
            f"  R²       : {r2:.5f}",
            f"  CV MAE   : {-cv.mean()*100:.2f} ± {cv.std()*100:.2f} cm  ({cv_k}-fold)\n",
        ]

        # Per-distance breakdown
        if df["true_dist_m"].nunique() > 1:
            txt.append("  Per-distance breakdown:")
            for d_val in sorted(df["true_dist_m"].unique()):
                mask = (df["true_dist_m"] - d_val).abs() < 0.01
                sX   = safe_build_X(df[mask], dist_feats)
                sy   = df.loc[mask, "true_dist_m"].values
                sp   = pipeline.predict(sX)
                rm   = float(np.mean(np.abs(sX[:, di] - sy))) if di is not None else float("nan")
                cm   = float(np.mean(np.abs(sp - sy)))
                txt.append(f"    {d_val:.2f} m : raw {rm*100:.1f} cm  →  "
                           f"corrected {cm*100:.1f} cm   ({mask.sum()} samples)")
            txt.append("")

        # Feature importance
        if algo in ("gradient_boosting", "random_forest"):
            imp = pipeline.named_steps["reg"].feature_importances_
            idx = np.argsort(imp)[::-1]
            txt.append("  Feature importance:")
            for i in idx:
                bar = "█" * max(1, int(imp[i] * 32))
                txt.append(f"    {dist_feats[i]:<22} {imp[i]:.4f}  {bar}")
            txt.append("")

        self.dist_model = pipeline
        self.dist_feats = dist_feats

        # ── Angle classifier ───────────────────────────────────────────────
        n_angles = df["angle_deg"].nunique() if "angle_deg" in df.columns else 0
        acc = None

        if self.train_angle_var.get() and n_angles >= 2 and angle_feats:
            try:
                Xa = safe_build_X(df, angle_feats)
            except ValueError as e:
                txt.append(f"Angle classifier SKIPPED: {e}\n")
                self.angle_model = None
            else:
                ya = df["angle_deg"].values.astype(int)
                Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
                    Xa, ya, test_size=0.2, random_state=42)
                clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", RandomForestClassifier(
                        n_estimators=400, max_depth=12, random_state=42))])
                clf.fit(Xa_tr, ya_tr)
                ya_pred = clf.predict(Xa_te)
                acc     = accuracy_score(ya_te, ya_pred)

                txt += [
                    f"ANGLE CLASSIFIER  (RandomForest)",
                    f"  Features : {', '.join(angle_feats)}",
                    f"  Samples  : {len(Xa_tr)} train / {len(Xa_te)} test",
                    f"  Classes  : {sorted(set(ya.tolist()))}°",
                    f"  Accuracy : {acc*100:.1f}%\n",
                ]
                self.angle_model = clf
                self.angle_feats = angle_feats
        else:
            reason = (f"need ≥ 2 angles in dataset, have {n_angles}"
                      if n_angles < 2
                      else "disabled by checkbox" if not self.train_angle_var.get()
                      else "no features selected")
            txt.append(f"Angle classifier: skipped  ({reason})\n")
            self.angle_model = None

        self.model_meta = {
            "dist_algo":    algo,
            "dist_feats":   dist_feats,
            "dist_mae_cm":  cor_mae * 100,
            "angle_feats":  angle_feats if self.angle_model else [],
            "angle_acc":    acc,
        }

        # ── Update result text ────────────────────────────────────────────
        self.train_results.config(state=tk.NORMAL)
        self.train_results.delete("1.0", tk.END)
        self.train_results.insert("1.0", "\n".join(txt))
        self.train_results.config(state=tk.DISABLED)

        status = f"Dist MAE {cor_mae*100:.1f} cm"
        if acc is not None:
            status += f"   |   Angle acc {acc*100:.0f}%"
        self.train_status.config(text=status, foreground="green")

        self._plot_training(X_te, y_te, y_pred, raw_te)

    def _plot_training(self, X_te, y_te, y_pred, raw_te):
        self._tr_fig.clear()
        ax1 = self._tr_fig.add_subplot(121)
        ax2 = self._tr_fig.add_subplot(122)

        re = raw_te - y_te
        ce = y_pred  - y_te

        # Raw vs corrected scatter
        ax1.scatter(y_te, raw_te, alpha=0.35, s=8, c="tomato",    label="Raw")
        ax1.scatter(y_te, y_pred, alpha=0.35, s=8, c="limegreen", label="Corrected")
        lo = min(y_te.min(), raw_te.min(), y_pred.min()) - 0.2
        hi = max(y_te.max(), raw_te.max(), y_pred.max()) + 0.2
        ax1.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="Perfect")
        ax1.set_xlabel("True (m)"); ax1.set_ylabel("Output (m)")
        ax1.set_title("Raw vs Corrected"); ax1.legend(fontsize=7)

        # Error histograms
        ax2.hist(re, bins=40, alpha=0.5, color="tomato",
                 label=f"Raw   σ={re.std():.3f} m")
        ax2.hist(ce, bins=40, alpha=0.5, color="limegreen",
                 label=f"Corr  σ={ce.std():.3f} m")
        ax2.axvline(0, color="k", ls="--", alpha=0.4)
        ax2.set_xlabel("Error (m)"); ax2.set_title("Error Distribution")
        ax2.legend(fontsize=7)

        self._tr_fig.tight_layout()
        self._tr_canvas.draw()

    def _save_models(self):
        if self.dist_model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".pkl", filetypes=[("Pickle", "*.pkl")],
            parent=self.root)
        if p:
            try:
                with open(p, "wb") as f:
                    pickle.dump({
                        "dist_model":  self.dist_model,
                        "dist_feats":  self.dist_feats,
                        "angle_model": self.angle_model,
                        "angle_feats": self.angle_feats,
                        "meta":        self.model_meta,
                    }, f)
                messagebox.showinfo("Saved", f"Models saved to {p}", parent=self.root)
            except Exception as e:
                messagebox.showerror("Save error", str(e), parent=self.root)

    def _load_models(self):
        p = filedialog.askopenfilename(
            filetypes=[("Pickle", "*.pkl")], parent=self.root)
        if not p:
            return
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            self.dist_model  = obj["dist_model"]
            self.dist_feats  = obj["dist_feats"]
            self.angle_model = obj.get("angle_model")
            self.angle_feats = obj.get("angle_feats", [])
            self.model_meta  = obj.get("meta", {})
            m = self.model_meta
            status = f"Loaded — Dist MAE {m.get('dist_mae_cm', 0):.1f} cm"
            if m.get("angle_acc") is not None:
                status += f"  |  Angle acc {m['angle_acc']*100:.0f}%"
            self.train_status.config(text=status, foreground="green")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    # ══════════════════════ LIVE INFERENCE ═════════════════════════════════

    def _toggle_inference(self):
        if self._infer_active:
            self._infer_active = False
            self.infer_btn.config(text="▶  Start Inference (live BLE)")
            self.infer_status.config(text="Inactive", foreground="gray")
        else:
            if self.dist_model is None:
                messagebox.showwarning("No Model",
                                       "Train or load a model first.")
                return
            self._infer_active = True
            self._infer_trail.clear()
            self._inf_raw_hist.clear()
            self._inf_corr_hist.clear()
            self._infer_pkt_seen = 0
            self._infer_started_at = time.time()
            self.infer_btn.config(text="■  Stop Inference")
            self.infer_status.config(
                text="Active — waiting for BLE packets…",
                foreground="#ffaa00")
            # If no packets arrive within ~5 s, surface a clearer hint
            self.root.after(5000, self._check_infer_traffic)

    def _check_infer_traffic(self):
        if not self._infer_active:
            return
        if self._infer_pkt_seen == 0:
            snap = self.engine.snapshot()
            connected = [n for n, d in snap.items() if d.get("connected")]
            if not connected:
                hint = "No devices connected — check the Connection panel."
            else:
                hint = (f"Connected to {','.join(connected)} but no T1 packets "
                        f"in 5 s — is T1 advertising and ranging?")
            self.infer_status.config(text=f"Active — {hint}", foreground="#ffaa00")
        else:
            self.infer_status.config(
                text=f"Active — {self._infer_pkt_seen} packets processed",
                foreground="green")

    def _update_inference(self, pkt: dict):
        """Apply the distance corrector (and optional angle classifier) to one packet."""
        if not self._infer_active or self.dist_model is None:
            return

        self._infer_pkt_seen += 1
        if self._infer_pkt_seen <= 3 or self._infer_pkt_seen % 25 == 0:
            self.infer_status.config(
                text=f"Active — {self._infer_pkt_seen} packets processed",
                foreground="green")

        # Build a single-row DataFrame with all raw + engineered features
        row = {k: pkt.get(k, float("nan")) for k in RAW_FEATURES}
        try:
            df1 = engineer(pd.DataFrame([row]))
        except Exception:
            return

        # Distance prediction
        try:
            Xd   = safe_build_X(df1, self.dist_feats)
            corr = float(self.dist_model.predict(Xd)[0])
        except Exception:
            corr = float(row.get("distance_m", float("nan")))

        raw = float(pkt.get("distance_m", float("nan")))
        self._inf_raw_hist.append(raw)
        self._inf_corr_hist.append(corr)

        # Angle prediction
        angle_est = None
        if self.angle_model is not None and self.angle_feats:
            try:
                Xa        = safe_build_X(df1, self.angle_feats)
                angle_est = int(self.angle_model.predict(Xa)[0])
            except Exception:
                pass

        # Position trail
        if angle_est is not None:
            rad = math.radians(angle_est)
            self._infer_trail.append((corr * math.cos(rad),
                                      corr * math.sin(rad)))

        # True distance/angle for error display
        try:
            true_d = float(self.infer_true_var.get())
            true_a = float(self.infer_true_angle_var.get())
        except ValueError:
            true_d = true_a = float("nan")

        err     = corr - true_d if not np.isnan(true_d) else float("nan")
        raw_err = raw  - true_d if not np.isnan(true_d) else float("nan")

        self.i_raw_lbl.config(text=f"{raw:.3f} m")
        self.i_corr_lbl.config(text=f"{corr:.3f} m")
        self.i_err_lbl.config(
            text=f"{err:+.3f} m" if not np.isnan(err) else "—",
            foreground=(
                "#00cc88"
                if (not np.isnan(err) and abs(err) < abs(raw_err))
                else "#ffaa00"))
        self.i_angle_lbl.config(
            text=f"{angle_est}°" if angle_est is not None else "—")

        self._redraw_inference(true_d, true_a, corr, angle_est)

    # ── OFFLINE REPLAY ─────────────────────────────────────────────────────
    def _replay_dataset(self):
        """Run the loaded model against the in-memory dataset and visualise."""
        if self.dist_model is None:
            messagebox.showwarning("No Model",
                "Train or load a model first (Train tab → Load…).")
            return
        if self.dataset.empty:
            messagebox.showwarning("No Dataset",
                "The in-memory dataset is empty. Collect, import, or load a CSV first.")
            return
        try:
            self._run_replay(self.dataset.copy(), label=DATASET_FILE)
        except Exception as e:
            messagebox.showerror("Replay error", str(e))

    def _replay_dataset_file(self):
        """Pick a CSV file and run the loaded model against it."""
        if self.dist_model is None:
            messagebox.showwarning("No Model",
                "Train or load a model first (Train tab → Load…).")
            return
        p = filedialog.askopenfilename(
            title="Pick a labelled CSV (must have true_dist_m + raw UWB cols)",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            parent=self.root)
        if not p:
            return
        try:
            df = pd.read_csv(p)
            self._run_replay(df, label=os.path.basename(p))
        except Exception as e:
            messagebox.showerror("Replay error", str(e))

    def _run_replay(self, df: pd.DataFrame, label: str):
        # If we were running live, pause it so the panel doesn't fight itself.
        if self._infer_active:
            self._infer_active = False
            self.infer_btn.config(text="▶  Start Inference (live BLE)")

        df = df.replace([np.inf, -np.inf], np.nan)
        if "true_dist_m" not in df.columns:
            messagebox.showerror("Replay error",
                "CSV is missing the true_dist_m column — replay needs labelled rows.")
            return
        df = df.dropna(subset=["true_dist_m", "distance_m"]).reset_index(drop=True)
        if df.empty:
            messagebox.showwarning("Empty after filter",
                "No usable rows (need both true_dist_m and distance_m).")
            return
        df = engineer(df)

        missing = [f for f in self.dist_feats if f not in df.columns]
        if missing:
            messagebox.showerror("Feature mismatch",
                f"CSV is missing model features: {missing}")
            return

        X = safe_build_X(df, self.dist_feats)
        y_true = df["true_dist_m"].astype(float).to_numpy()
        raw    = df["distance_m"].astype(float).to_numpy()
        y_pred = np.asarray(self.dist_model.predict(X), dtype=float)

        raw_mae  = float(np.mean(np.abs(raw - y_true)))
        corr_mae = float(np.mean(np.abs(y_pred - y_true)))
        improv = (raw_mae - corr_mae) / raw_mae * 100 if raw_mae > 0 else 0.0

        # Update the four metric labels with summary stats
        self.i_raw_lbl.config(text=f"{raw_mae*100:.2f} cm")
        self.i_corr_lbl.config(text=f"{corr_mae*100:.2f} cm")
        self.i_err_lbl.config(
            text=f"−{improv:.1f} %" if improv >= 0 else f"+{-improv:.1f} %",
            foreground=("#00cc88" if improv > 0 else "#ffaa00"))
        self.i_angle_lbl.config(text=f"n={len(df)}")

        self.infer_status.config(
            text=(f"Replay {label}: {len(df)} rows  ·  "
                  f"raw MAE {raw_mae*100:.1f} cm  →  corr MAE {corr_mae*100:.1f} cm  "
                  f"({improv:+.1f}% improvement)"),
            foreground="#00cc88" if improv > 0 else "#ffaa00")

        # Repurpose the two existing matplotlib axes for the offline view
        fig = self._inf_fig
        ax_left  = self._inf_ax_pos
        ax_right = self._inf_ax_dist
        ax_left.clear(); ax_right.clear()
        for ax in (ax_left, ax_right):
            ax.set_facecolor("#0d0d0d")
            for s in ax.spines.values():
                s.set_color("#888")
            ax.tick_params(colors="#bbb")
            ax.title.set_color("#ddd")
            ax.xaxis.label.set_color("#bbb")
            ax.yaxis.label.set_color("#bbb")

        ax_left.scatter(y_true, raw,    s=8, alpha=0.45, c="tomato",
                        label=f"raw   (MAE {raw_mae*100:.1f} cm)")
        ax_left.scatter(y_true, y_pred, s=8, alpha=0.55, c="#00cc88",
                        label=f"corr  (MAE {corr_mae*100:.1f} cm)")
        lo = float(min(y_true.min(), raw.min(), y_pred.min())) - 0.2
        hi = float(max(y_true.max(), raw.max(), y_pred.max())) + 0.2
        ax_left.plot([lo, hi], [lo, hi], "--", color="#888", alpha=0.7,
                     label="y = x (perfect)")
        ax_left.set_xlabel("true distance (m)")
        ax_left.set_ylabel("model output (m)")
        ax_left.set_title(f"Replay: {label}  ·  n={len(df)}")
        ax_left.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#444",
                       labelcolor="#ddd")

        ax_right.hist(raw - y_true, bins=40, color="tomato", alpha=0.55,
                      label=f"raw   σ={(raw-y_true).std():.2f}")
        ax_right.hist(y_pred - y_true, bins=40, color="#00cc88", alpha=0.55,
                      label=f"corr  σ={(y_pred-y_true).std():.2f}")
        ax_right.axvline(0, color="#bbb", ls="--", alpha=0.5)
        ax_right.set_xlabel("error (m)")
        ax_right.set_title("Error distribution")
        ax_right.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#444",
                        labelcolor="#ddd")

        fig.tight_layout()
        self._inf_canvas.draw_idle()

    def _redraw_inference(self, true_d, true_a, corr, angle_est):
        fig     = self._inf_fig
        ax_pos  = self._inf_ax_pos
        ax_dist = self._inf_ax_dist
        ax_pos.clear()
        ax_dist.clear()

        # ── Position map ──────────────────────────────────────────────────
        max_r = max(corr * 1.5, 1.0)

        # Range rings
        theta = np.linspace(0, 2 * np.pi, 200)
        ring_vals = [v for v in [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10]
                     if v < max_r * 1.15]
        for r in ring_vals:
            ax_pos.plot(r * np.cos(theta), r * np.sin(theta),
                        ls="--", color="#333355", lw=0.6)
            ax_pos.text(r * 0.72, r * 0.72, f"{r}m",
                        fontsize=6, color="#555577")

        # Corrected distance ring (highlight)
        ax_pos.plot(corr * np.cos(theta), corr * np.sin(theta),
                    ls="--", color="#006688", lw=1.2, alpha=0.7)

        # Anchor at origin
        ax_pos.plot(0, 0, "D", color="lime", ms=10, zorder=5)
        ax_pos.text(0.05, 0.1, "A1", color="lime", fontsize=8)

        # Position trail
        trail = list(self._infer_trail)
        if len(trail) > 1:
            xs = [p[0] for p in trail[:-1]]
            ys = [p[1] for p in trail[:-1]]
            ax_pos.plot(xs, ys, "-", color="#336688", alpha=0.35, lw=0.8)

        # Current estimated position
        if angle_est is not None:
            rad = math.radians(angle_est)
            ex, ey = corr * math.cos(rad), corr * math.sin(rad)
            ax_pos.plot(ex, ey, "o", color="#00ddff", ms=11, zorder=6,
                        label=f"Est  {corr:.2f} m @ {angle_est}°")
            # Direction line from anchor
            ax_pos.plot([0, ex * 0.9], [0, ey * 0.9],
                        "-", color="#00ddff", alpha=0.4, lw=1)
        else:
            # No angle — just show the range circle label
            ax_pos.text(corr * 0.05, corr + 0.12, f"{corr:.2f} m",
                        color="#00ddff", fontsize=8)

        # True position
        if not (np.isnan(true_d) or np.isnan(true_a)):
            rad_t = math.radians(true_a)
            tx, ty = true_d * math.cos(rad_t), true_d * math.sin(rad_t)
            ax_pos.plot(tx, ty, "x", color="orange", ms=13, mew=2.5, zorder=7,
                        label=f"True {true_d:.2f} m @ {true_a:.0f}°")

        ax_pos.set_xlim(-max_r, max_r)
        ax_pos.set_ylim(-max_r, max_r)
        ax_pos.set_aspect("equal")
        ax_pos.set_facecolor("#0a0a18")
        ax_pos.set_title("Position Estimate", color="#cccccc", fontsize=9)
        ax_pos.tick_params(colors="#666688", labelsize=7)
        for sp in ax_pos.spines.values():
            sp.set_color("#333355")
        if angle_est is not None or not np.isnan(true_d):
            ax_pos.legend(fontsize=7, loc="upper right",
                          facecolor="#111122", labelcolor="white")

        # ── Distance time series ──────────────────────────────────────────
        xs = range(len(self._inf_raw_hist))
        if self._inf_raw_hist:
            ax_dist.plot(list(xs), list(self._inf_raw_hist),
                         color="tomato", alpha=0.55, lw=0.9, label="Raw")
            ax_dist.plot(list(xs), list(self._inf_corr_hist),
                         color="limegreen", alpha=0.9, lw=1.1, label="Corrected")
        if not np.isnan(true_d):
            ax_dist.axhline(true_d, color="orange", ls="--", lw=1.0,
                            label=f"True {true_d:.2f} m")
        ax_dist.set_xlabel("Sample", fontsize=8)
        ax_dist.set_ylabel("Distance (m)", fontsize=8)
        ax_dist.set_title("Live Distance", fontsize=9)
        ax_dist.legend(fontsize=7, loc="upper right")
        ax_dist.tick_params(labelsize=7)

        fig.tight_layout()
        self._inf_canvas.draw_idle()

    # ══════════════════════ CLEANUP ════════════════════════════════════════

    def on_close(self):
        # Routed through _on_close which already shuts the engine + logger.
        self._collecting   = False
        self._infer_active = False
        self._on_close()


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def main():
    if not HAS_BLEAK:
        print(
            "Warning: 'bleak' library not found.\n"
            "  Direct BLE will not work — run:  pip install bleak\n"
        )
    root = tk.Tk()
    UWBBLECalApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
