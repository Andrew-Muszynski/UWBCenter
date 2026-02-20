#!/usr/bin/env python3
"""
uwb_ble_calibration.py — BLE-based UWB Calibration & ML Tool
=============================================================
Works alongside uwb_dashboard.py (must be running at localhost:5000).

Workflow:
  1. Collect  — label live BLE packets with known distance + orientation angle
  2. Import   — bulk-load existing logs/tag_data_*.csv files with labels
  3. Dataset  — view and manage accumulated calibration data
  4. Train    — fit a distance corrector (regression) and optional angle
                classifier (random forest) on the labelled data
  5. Infer    — apply the trained model to the live BLE stream and show a
                corrected position on a 2-D polar map

Requirements:
    pip install requests scikit-learn numpy pandas matplotlib
    (tkinter is part of the Python standard library)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import threading
import time
import os
import math
import pickle

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import deque

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

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

DASHBOARD_URL  = "http://localhost:5000/api/state"
POLL_INTERVAL  = 0.5          # seconds between /api/state polls
DATASET_FILE   = "ble_cal_dataset.csv"
ANGLE_CHOICES  = ["0", "45", "90", "135", "180", "225", "270", "315"]

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


# ═══════════════════════════════ BLE POLLER ═══════════════════════════════════

class BLEPoller:
    """
    Background thread that polls /api/state and dispatches only NEW tag packets
    (deduplicated by seq number) to registered callbacks.
    """

    def __init__(self):
        self._callbacks  = []
        self._running    = False
        self._thread     = None
        self._max_seq    = {}     # device_name → highest seq seen

    def subscribe(self, cb):
        self._callbacks.append(cb)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def reset_seq(self):
        """Call before a new collection session to replay any buffered packets."""
        self._max_seq.clear()

    def _loop(self):
        while self._running:
            self._poll()
            time.sleep(POLL_INTERVAL)

    def _poll(self):
        if not HAS_REQUESTS:
            return
        try:
            resp = requests.get(DASHBOARD_URL, timeout=3)
            data = resp.json()
        except Exception:
            return

        for name, dev in data.get("devices", {}).items():
            if dev.get("type") != "tag":
                continue
            max_seen = self._max_seq.get(name, -1)
            new_pkts = sorted(
                [p for p in dev.get("history", []) if p.get("seq", -1) > max_seen],
                key=lambda p: p.get("seq", 0),
            )
            for pkt in new_pkts:
                s = pkt.get("seq", 0)
                if s > max_seen:
                    max_seen = s
                for cb in self._callbacks:
                    try:
                        cb(name, pkt)
                    except Exception:
                        pass
            if new_pkts:
                self._max_seq[name] = max_seen


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

        # ── BLE poller ─────────────────────────────────────────────────────
        self.poller = BLEPoller()
        self.poller.subscribe(self._on_packet)
        self.poller.start()

        self._build_ui()
        self._load_dataset()

    # ══════════════════════════════ UI CONSTRUCTION ════════════════════════

    def _build_ui(self):
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

    # ── Tab 1: Collect ─────────────────────────────────────────────────────

    def _build_collect_tab(self):
        t = self.t_collect

        # Dashboard connection
        sf = ttk.LabelFrame(t, text="Dashboard Connection", padding=8)
        sf.pack(fill=tk.X, padx=10, pady=(10, 4))
        sr = ttk.Frame(sf); sr.pack(fill=tk.X)
        ttk.Label(sr, text="URL:").pack(side=tk.LEFT, padx=(0, 4))
        self.url_var = tk.StringVar(value=DASHBOARD_URL)
        ttk.Entry(sr, textvariable=self.url_var, width=42).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(sr, text="Test Connection",
                   command=self._test_connection).pack(side=tk.LEFT, padx=(0, 10))
        self.conn_lbl = ttk.Label(sr, text="Unknown", foreground="gray")
        self.conn_lbl.pack(side=tk.LEFT)

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

        info = ttk.LabelFrame(t, text="Import Existing tag_data_*.csv Logs", padding=10)
        info.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(info, wraplength=900,
            text=(
                "Select one or more tag CSV files from logs/.  "
                "Each file will be labelled with the true distance and angle you enter.  "
                "If 'Ask per file' is checked you will be prompted for each file individually, "
                "which is useful when a single run contains only one configuration."
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
        self.ds_sum = tk.Text(sf, height=8, font=("Consolas", 9), state=tk.DISABLED)
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
                                     state=tk.DISABLED, wrap=tk.WORD)
        self.train_results.pack(fill=tk.X, pady=(0, 4))
        self._tr_fig    = Figure(figsize=(7, 4), dpi=96)
        self._tr_canvas = FigureCanvasTkAgg(self._tr_fig, master=rf)
        self._tr_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Tab 5: Live Inference ───────────────────────────────────────────────

    def _build_infer_tab(self):
        t = self.t_infer

        ctrl = ttk.Frame(t); ctrl.pack(fill=tk.X, padx=10, pady=8)
        self.infer_btn = ttk.Button(ctrl, text="▶  Start Inference",
                                    command=self._toggle_inference)
        self.infer_btn.pack(side=tk.LEFT, padx=(0, 12))

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

    # ══════════════════════ CONNECTION TEST ════════════════════════════════

    def _test_connection(self):
        if not HAS_REQUESTS:
            self.conn_lbl.config(
                text="'requests' not installed — run: pip install requests",
                foreground="red")
            return
        try:
            r    = requests.get(self.url_var.get(), timeout=3)
            data = r.json()
            devs = list(data.get("devices", {}).keys())
            self.conn_lbl.config(
                text=f"✓  Connected  —  devices: {', '.join(devs) or 'none found'}",
                foreground="green")
        except Exception as e:
            self.conn_lbl.config(text=f"✗  Failed: {e}", foreground="red")

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
            title="Select tag_data_*.csv files",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            initialdir=str(log_dir))
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

        # Determine distance/angle label
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
        p = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            self.dataset.to_csv(p, index=False)
            messagebox.showinfo("Saved", f"Exported {len(self.dataset)} rows to {p}")

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

        df = engineer(
            self.dataset.dropna(subset=["distance_m", "true_dist_m"]).copy()
        )
        dist_feats  = self._get_feat_list(self.dist_feat_vars)
        angle_feats = self._get_feat_list(self.angle_feat_vars)

        if not dist_feats:
            messagebox.showwarning("No Features",
                                   "Select at least one distance feature.")
            return

        algo = self.dist_algo_var.get()
        txt  = [f"{'='*58}", "  ML TRAINING REPORT", f"{'='*58}\n"]

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
            defaultextension=".pkl", filetypes=[("Pickle", "*.pkl")])
        if p:
            with open(p, "wb") as f:
                pickle.dump({
                    "dist_model":  self.dist_model,
                    "dist_feats":  self.dist_feats,
                    "angle_model": self.angle_model,
                    "angle_feats": self.angle_feats,
                    "meta":        self.model_meta,
                }, f)
            messagebox.showinfo("Saved", f"Models saved to {p}")

    def _load_models(self):
        p = filedialog.askopenfilename(filetypes=[("Pickle", "*.pkl")])
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
            self.infer_btn.config(text="▶  Start Inference")
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
            self.infer_btn.config(text="■  Stop Inference")
            self.infer_status.config(text="Active", foreground="green")

    def _update_inference(self, pkt: dict):
        """Apply the distance corrector (and optional angle classifier) to one packet."""
        if not self._infer_active or self.dist_model is None:
            return

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
        self.poller.stop()
        self._collecting   = False
        self._infer_active = False
        self.root.destroy()


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def main():
    if not HAS_REQUESTS:
        print(
            "Warning: 'requests' library not found.\n"
            "  Live BLE polling will not work.\n"
            "  Install with:  pip install requests\n"
        )
    root = tk.Tk()
    app  = UWBBLECalApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
