"""
DWM1000 UWB Ranging Calibration & ML Correction Tool
=====================================================
Collects TWR ranging data from Arduino DWM1000 initiator over serial,
logs all DWM1000 diagnostic features, and trains ML models to correct
systematic ranging errors using signal quality metrics.

Matches Arduino CSV format:
  sample, millis, distance_m, round_trip_ticks, reply_delay_ticks,
  tof_ticks, rx_power_dBm, fp_power_dBm, quality, temp_C, pressure_hPa

Arduino serial commands supported:
  start [N]     - begin collecting N samples
  stop          - abort collection
  delay XXXXX   - set antenna delay register
  distance X.XX - set known reference distance
  status        - print current settings
  env           - read environmental sensors

Requirements:
    pip install pyserial scikit-learn numpy pandas matplotlib
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import serial
import serial.tools.list_ports
import threading
import time
import re
import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import (
    train_test_split, cross_val_score, LeaveOneGroupOut, GroupKFold,
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier,
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report,
)
from sklearn.pipeline import Pipeline
import pickle

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ── Column definitions matching Arduino CSV output ─────────────────────
CSV_COLUMNS = [
    "sample", "millis", "distance_m", "round_trip_ticks", "reply_delay_ticks",
    "tof_ticks", "rx_power_dBm", "fp_power_dBm", "quality", "temp_C", "pressure_hPa"
]

# ── ML features (selectable in UI) ────────────────────────────────────
ML_FEATURE_DEFS = {
    "distance_m":        "Raw measured distance (m)",
    "tof_ticks":         "Time-of-flight ticks",
    "rx_power_dBm":      "Received signal power (dBm)",
    "fp_power_dBm":      "First-path power (dBm)",
    "quality":           "Signal quality index",
    "power_diff_dB":     "rx_power - fp_power (multipath indicator)",
    "temp_C":            "Temperature (C)",
    "pressure_hPa":      "Barometric pressure (hPa)",
    "round_trip_ticks":  "Round-trip ticks",
    "reply_delay_ticks": "Reply delay ticks",
    "channel_condition":  "NLOS class (0=LOS, 1=marginal, 2=NLOS)",
    "confidence":        "Measurement confidence (0 to 1)",
}

DEFAULT_FEATURES = [
    "distance_m", "tof_ticks", "rx_power_dBm",
    "fp_power_dBm", "quality", "power_diff_dB",
    "channel_condition", "confidence",
]

# ── NLOS classification thresholds (DW1000 User Manual Section 4.7) ───
# power_diff_dB = rx_power_dBm - fp_power_dBm
#   < 6 dB  => LOS (first path carries most energy)
#   6-10 dB => Marginal (some multipath)
#   > 10 dB => NLOS (heavy multipath / obstruction)
NLOS_THRESH_LOS      = 6.0   # below this = LOS
NLOS_THRESH_NLOS     = 10.0  # above this = NLOS


def classify_channel(power_diff_dB):
    """Classify channel condition from rx_power - fp_power difference.
    Returns: 0 = LOS, 1 = Marginal, 2 = NLOS"""
    if hasattr(power_diff_dB, '__iter__'):
        return np.where(
            power_diff_dB < NLOS_THRESH_LOS, 0,
            np.where(power_diff_dB > NLOS_THRESH_NLOS, 2, 1)
        ).astype(float)
    if power_diff_dB < NLOS_THRESH_LOS:
        return 0.0
    elif power_diff_dB > NLOS_THRESH_NLOS:
        return 2.0
    return 1.0


def compute_confidence(quality, power_diff_dB, distance_std=None):
    """Compute a 0-1 confidence score for a measurement.

    Combines:
      - quality: higher is better (normalized via sigmoid)
      - power_diff_dB: lower is better (LOS has small diff)
      - distance_std: rolling std, lower is better (optional)
    """
    # Quality component: sigmoid centered around typical quality value
    q = np.clip(quality, 0, 1000)
    q_score = 1.0 / (1.0 + np.exp(-(q - 150) / 50))

    # Power diff component: LOS (< 6 dB) gets high score
    pd = np.clip(power_diff_dB, 0, 30)
    pd_score = np.clip(1.0 - pd / 20.0, 0, 1)

    # Combine (weighted)
    conf = 0.6 * q_score + 0.4 * pd_score

    # Optional: penalize high-variance measurements
    if distance_std is not None:
        std_penalty = np.clip(1.0 - distance_std / 0.5, 0, 1)
        conf = conf * 0.7 + std_penalty * 0.3

    return np.clip(conf, 0, 1)


class UWBCalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DWM1000 UWB Calibration & ML Correction")
        self.root.geometry("1280x920")
        self.root.minsize(1050, 780)

        # ── Serial state ──
        self.serial_conn = None
        self.serial_thread = None
        self.serial_running = False

        # ── Collection state ──
        self.collecting = False
        self.sample_count = 0
        self.session_buffer = []
        self.live_readings = []

        # ── Master dataset ──
        self.dataset = pd.DataFrame()
        self.dataset_path = "uwb_dataset.csv"

        # ── ML model ──
        self.model = None
        self.model_features = []
        self.model_info = {}
        self.nlos_model = None

        # ── Live correction ──
        self.live_correcting = False
        self.live_raw_history = []
        self.live_corr_history = []

        self._build_ui()
        self._load_existing_dataset()
        self._refresh_ports()

    # ==================================================================
    #                        UI CONSTRUCTION
    # ==================================================================

    def _build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_collect = ttk.Frame(self.notebook)
        self.tab_dataset = ttk.Frame(self.notebook)
        self.tab_ml      = ttk.Frame(self.notebook)
        self.tab_live    = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_collect, text="  Data Collection  ")
        self.notebook.add(self.tab_dataset, text="  Dataset  ")
        self.notebook.add(self.tab_ml,      text="  ML Training  ")
        self.notebook.add(self.tab_live,    text="  Live Correction  ")

        self._build_collection_tab()
        self._build_dataset_tab()
        self._build_ml_tab()
        self._build_live_tab()

    # ────────────── Tab 1: Data Collection ─────────────────────────────

    def _build_collection_tab(self):
        tab = self.tab_collect

        # -- Serial Connection --
        conn = ttk.LabelFrame(tab, text="Serial Connection", padding=10)
        conn.pack(fill=tk.X, padx=10, pady=(10, 5))
        row = ttk.Frame(conn); row.pack(fill=tk.X)

        ttk.Label(row, text="Port:").pack(side=tk.LEFT, padx=(0, 5))
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(row, textvariable=self.port_var,
                                       width=20, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row, text="Refresh", command=self._refresh_ports
                   ).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(row, text="Baud:").pack(side=tk.LEFT, padx=(0, 5))
        self.baud_var = tk.StringVar(value="115200")
        ttk.Combobox(row, textvariable=self.baud_var, width=10,
                     values=["9600","57600","115200","230400"],
                     state="readonly").pack(side=tk.LEFT, padx=(0, 15))

        self.connect_btn = ttk.Button(row, text="Connect",
                                      command=self._toggle_serial)
        self.connect_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.conn_status = ttk.Label(row, text="Disconnected", foreground="red")
        self.conn_status.pack(side=tk.LEFT)

        # -- Test Configuration --
        cfg = ttk.LabelFrame(tab, text="Test Configuration", padding=10)
        cfg.pack(fill=tk.X, padx=10, pady=5)
        r1 = ttk.Frame(cfg); r1.pack(fill=tk.X, pady=2)

        ttk.Label(r1, text="True Distance (m):").pack(side=tk.LEFT, padx=(0,5))
        self.true_dist_var = tk.StringVar(value="1.00")
        ttk.Entry(r1, textvariable=self.true_dist_var, width=10
                  ).pack(side=tk.LEFT, padx=(0,20))

        ttk.Label(r1, text="Angle (deg):").pack(side=tk.LEFT, padx=(0,5))
        self.angle_var = tk.StringVar(value="0")
        ttk.Combobox(r1, textvariable=self.angle_var, width=8,
                     values=["0","90","180","270"]
                     ).pack(side=tk.LEFT, padx=(0,20))

        ttk.Label(r1, text="Antenna Delay:").pack(side=tk.LEFT, padx=(0,5))
        self.ant_delay_var = tk.StringVar(value="0")
        ttk.Entry(r1, textvariable=self.ant_delay_var, width=10
                  ).pack(side=tk.LEFT, padx=(0,20))

        ttk.Label(r1, text="Notes:").pack(side=tk.LEFT, padx=(0,5))
        self.notes_var = tk.StringVar()
        ttk.Entry(r1, textvariable=self.notes_var, width=20
                  ).pack(side=tk.LEFT)

        # -- Format info --
        fmt = ttk.LabelFrame(tab, text="Serial Parse Format", padding=8)
        fmt.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(fmt, font=("Consolas", 9),
            text="Expected: sample,millis,distance_m,round_trip_ticks,"
                 "reply_delay_ticks,tof_ticks,rx_power_dBm,fp_power_dBm,"
                 "quality,temp_C,pressure_hPa"
        ).pack(anchor=tk.W)
        r2 = ttk.Frame(fmt); r2.pack(fill=tk.X, pady=3)
        ttk.Label(r2, text="Lines starting with '#' or non-numeric are ignored."
                  ).pack(side=tk.LEFT)
        self.auto_start_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r2, text="Auto-send 'distance' + 'start' on collection",
                        variable=self.auto_start_var).pack(side=tk.RIGHT)

        # -- Collection Controls --
        ctl = ttk.LabelFrame(tab, text="Collection Controls", padding=10)
        ctl.pack(fill=tk.X, padx=10, pady=5)
        cr = ttk.Frame(ctl); cr.pack(fill=tk.X)

        self.start_btn = ttk.Button(cr, text="Start Collection",
                                    command=self._start_collection)
        self.start_btn.pack(side=tk.LEFT, padx=(0,10))
        self.stop_btn = ttk.Button(cr, text="Stop Collection",
                                   command=self._stop_collection,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0,20))

        self.sample_lbl = ttk.Label(cr, text="Samples: 0",
                                    font=("Helvetica", 12, "bold"))
        self.sample_lbl.pack(side=tk.LEFT, padx=(0,20))
        self.stats_lbl = ttk.Label(cr, text="Mean: --  |  Std: --  |  Error: --")
        self.stats_lbl.pack(side=tk.LEFT)

        # -- Send command row --
        sr = ttk.Frame(ctl); sr.pack(fill=tk.X, pady=(5,0))
        ttk.Label(sr, text="Send:").pack(side=tk.LEFT, padx=(0,5))
        self.send_var = tk.StringVar()
        ent = ttk.Entry(sr, textvariable=self.send_var, width=40)
        ent.pack(side=tk.LEFT, padx=(0,5))
        ent.bind("<Return>", lambda e: self._send_cmd())
        ttk.Button(sr, text="Send", command=self._send_cmd).pack(side=tk.LEFT)

        # Quick buttons for common Arduino commands
        qr = ttk.Frame(ctl); qr.pack(fill=tk.X, pady=(5,0))
        ttk.Label(qr, text="Quick:").pack(side=tk.LEFT, padx=(0,5))
        for label, cmd in [("status","status"), ("env","env"),
                           ("stop","stop")]:
            ttk.Button(qr, text=label, width=8,
                       command=lambda c=cmd: self._quick_send(c)
                       ).pack(side=tk.LEFT, padx=2)

        # -- Serial Monitor --
        mon = ttk.LabelFrame(tab, text="Serial Monitor", padding=5)
        mon.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5,10))
        self.serial_text = tk.Text(mon, height=10, bg="#1e1e1e", fg="#00ff00",
                                   font=("Consolas", 9), state=tk.DISABLED)
        sb = ttk.Scrollbar(mon, command=self.serial_text.yview)
        self.serial_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.serial_text.pack(fill=tk.BOTH, expand=True)

    # ────────────── Tab 2: Dataset ─────────────────────────────────────

    def _build_dataset_tab(self):
        tab = self.tab_dataset

        ctl = ttk.Frame(tab); ctl.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(ctl, text="Refresh",
                   command=self._refresh_dataset_view).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctl, text="Import CSV / TXT",
                   command=self._import_file).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctl, text="Export CSV",
                   command=self._export_csv).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctl, text="Clear Dataset",
                   command=self._clear_dataset).pack(side=tk.LEFT, padx=3)
        self.ds_info = ttk.Label(ctl, text="Dataset: 0 samples")
        self.ds_info.pack(side=tk.RIGHT, padx=10)

        # Treeview
        tf = ttk.Frame(tab)
        tf.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,5))
        cols = ("true_dist","distance_m","error","tof_ticks",
                "rx_power","fp_power","quality","angle","temp","pressure")
        self.ds_tree = ttk.Treeview(tf, columns=cols, show="headings", height=14)
        for c, h, w in [
            ("true_dist","True(m)",75), ("distance_m","Meas(m)",85),
            ("error","Error(m)",80), ("tof_ticks","ToF Ticks",80),
            ("rx_power","RxPwr",65), ("fp_power","FPPwr",65),
            ("quality","Quality",65), ("angle","Angle",55),
            ("temp","Temp",55), ("pressure","hPa",60)]:
            self.ds_tree.heading(c, text=h)
            self.ds_tree.column(c, width=w, anchor=tk.CENTER)
        vs = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=self.ds_tree.yview)
        self.ds_tree.configure(yscrollcommand=vs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        self.ds_tree.pack(fill=tk.BOTH, expand=True)

        # Summary
        sf = ttk.LabelFrame(tab, text="Summary by Configuration", padding=8)
        sf.pack(fill=tk.X, padx=10, pady=(0,10))
        self.summary_text = tk.Text(sf, height=10, font=("Consolas", 9),
                                    state=tk.DISABLED)
        self.summary_text.pack(fill=tk.X)

    # ────────────── Tab 3: ML Training ─────────────────────────────────

    def _build_ml_tab(self):
        tab = self.tab_ml

        cfg = ttk.LabelFrame(tab, text="Model Configuration", padding=10)
        cfg.pack(fill=tk.X, padx=10, pady=(10,5))

        r1 = ttk.Frame(cfg); r1.pack(fill=tk.X, pady=3)
        ttk.Label(r1, text="Algorithm:").pack(side=tk.LEFT, padx=(0,5))
        self.algo_var = tk.StringVar(value="gradient_boosting")
        ttk.Combobox(r1, textvariable=self.algo_var, width=25, state="readonly",
                     values=["polynomial_regression","ridge_polynomial",
                             "gradient_boosting","random_forest"]
                     ).pack(side=tk.LEFT, padx=(0,20))

        ttk.Label(r1, text="Poly Degree:").pack(side=tk.LEFT, padx=(0,5))
        self.poly_var = tk.StringVar(value="3")
        ttk.Spinbox(r1, textvariable=self.poly_var, from_=1, to=6, width=5
                    ).pack(side=tk.LEFT, padx=(0,20))

        ttk.Label(r1, text="Test Split:").pack(side=tk.LEFT, padx=(0,5))
        self.split_var = tk.StringVar(value="0.2")
        ttk.Entry(r1, textvariable=self.split_var, width=6).pack(side=tk.LEFT)

        # Pipeline options row
        r1b = ttk.Frame(cfg); r1b.pack(fill=tk.X, pady=3)
        self.multistage_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r1b, text="Multi-stage pipeline (NLOS classifier + regressor)",
                        variable=self.multistage_var).pack(side=tk.LEFT, padx=(0,20))
        self.lodo_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r1b, text="LODO cross-validation (leave-one-distance-out)",
                        variable=self.lodo_var).pack(side=tk.LEFT)

        # Feature checkboxes
        ff = ttk.LabelFrame(cfg, text="Feature Selection", padding=8)
        ff.pack(fill=tk.X, pady=5)
        self.feat_vars = {}
        fg = ttk.Frame(ff); fg.pack(fill=tk.X)
        for i, (key, desc) in enumerate(ML_FEATURE_DEFS.items()):
            v = tk.BooleanVar(value=(key in DEFAULT_FEATURES))
            self.feat_vars[key] = v
            ttk.Checkbutton(fg, text=f"{key}  ({desc})", variable=v
                            ).grid(row=i//2, column=i%2, sticky=tk.W, padx=10, pady=1)

        br = ttk.Frame(cfg); br.pack(fill=tk.X, pady=5)
        ttk.Button(br, text="Train Model",
                   command=self._train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(br, text="Save Model",
                   command=self._save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(br, text="Load Model",
                   command=self._load_model).pack(side=tk.LEFT, padx=5)
        self.model_status = ttk.Label(br, text="No model trained",
                                      foreground="gray")
        self.model_status.pack(side=tk.LEFT, padx=20)

        # Results area
        rf = ttk.LabelFrame(tab, text="Training Results", padding=5)
        rf.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        paned = ttk.PanedWindow(rf, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        txf = ttk.Frame(paned); paned.add(txf, weight=1)
        self.results_text = tk.Text(txf, font=("Consolas", 9),
                                    state=tk.DISABLED, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        pf = ttk.Frame(paned); paned.add(pf, weight=2)
        self.fig = Figure(figsize=(7,5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ────────────── Tab 4: Live Correction ─────────────────────────────

    def _build_live_tab(self):
        tab = self.tab_live

        info = ttk.LabelFrame(tab, text="Live Correction Mode", padding=10)
        info.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(info, wraplength=700,
            text="Reads live serial data and applies the trained ML model "
                 "for real-time correction. Serial connection must be active "
                 "and a model must be trained/loaded."
        ).pack(anchor=tk.W)

        cr = ttk.Frame(info); cr.pack(fill=tk.X, pady=8)
        ttk.Label(cr, text="True Distance (m):").pack(side=tk.LEFT, padx=(0,5))
        self.live_true_var = tk.StringVar(value="1.00")
        ttk.Entry(cr, textvariable=self.live_true_var, width=10
                  ).pack(side=tk.LEFT, padx=(0,15))
        ttk.Label(cr, text="Angle:").pack(side=tk.LEFT, padx=(0,5))
        self.live_angle_var = tk.StringVar(value="0")
        ttk.Combobox(cr, textvariable=self.live_angle_var, width=8,
                     values=["0","90","180","270"]
                     ).pack(side=tk.LEFT, padx=(0,15))
        self.live_btn = ttk.Button(cr, text="Start Live Correction",
                                   command=self._toggle_live)
        self.live_btn.pack(side=tk.LEFT)

        # Big number displays
        nf = ttk.Frame(tab); nf.pack(fill=tk.X, padx=10, pady=5)

        raw_f = ttk.LabelFrame(nf, text="Raw Reading", padding=12)
        raw_f.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.raw_val = ttk.Label(raw_f, text="-- m",
                                 font=("Helvetica",26,"bold"))
        self.raw_val.pack()
        self.raw_err = ttk.Label(raw_f, text="Error: --",
                                 font=("Helvetica",11))
        self.raw_err.pack()

        cor_f = ttk.LabelFrame(nf, text="ML Corrected", padding=12)
        cor_f.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.cor_val = ttk.Label(cor_f, text="-- m",
                                 font=("Helvetica",26,"bold"), foreground="green")
        self.cor_val.pack()
        self.cor_err = ttk.Label(cor_f, text="Error: --",
                                 font=("Helvetica",11))
        self.cor_err.pack()

        st_f = ttk.LabelFrame(nf, text="Rolling Stats (last 50)", padding=12)
        st_f.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.roll_raw = ttk.Label(st_f, text="Raw  u: --  s: --",
                                  font=("Consolas",10))
        self.roll_raw.pack(anchor=tk.W)
        self.roll_cor = ttk.Label(st_f, text="Corr u: --  s: --",
                                  font=("Consolas",10))
        self.roll_cor.pack(anchor=tk.W)
        self.roll_mae = ttk.Label(st_f, text="Raw MAE: --  Corr MAE: --",
                                  font=("Consolas",10))
        self.roll_mae.pack(anchor=tk.W)

        # Live plot
        lpf = ttk.LabelFrame(tab, text="Live Readings", padding=5)
        lpf.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        self.live_fig = Figure(figsize=(8,3), dpi=100)
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, master=lpf)
        self.live_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ==================================================================
    #                       SERIAL HANDLING
    # ==================================================================

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)

    def _toggle_serial(self):
        if self.serial_conn and self.serial_conn.is_open:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.port_var.get()
        baud = int(self.baud_var.get())
        if not port:
            messagebox.showwarning("No Port", "Select a serial port.")
            return
        try:
            self.serial_conn = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # Arduino reset
            self.serial_running = True
            self.serial_thread = threading.Thread(target=self._reader,
                                                  daemon=True)
            self.serial_thread.start()
            self.connect_btn.config(text="Disconnect")
            self.conn_status.config(text=f"Connected ({port})",
                                    foreground="green")
            self._log(f"Connected to {port} @ {baud}\n")
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))

    def _disconnect(self):
        self.serial_running = False
        if self.collecting:
            self._stop_collection()
        if self.serial_conn:
            try: self.serial_conn.close()
            except: pass
        self.serial_conn = None
        self.connect_btn.config(text="Connect")
        self.conn_status.config(text="Disconnected", foreground="red")
        self._log("Disconnected\n")

    def _reader(self):
        """Background thread: read serial lines."""
        while self.serial_running:
            try:
                if (self.serial_conn and self.serial_conn.is_open
                        and self.serial_conn.in_waiting):
                    line = self.serial_conn.readline().decode(
                        'utf-8', errors='replace').strip()
                    if line:
                        self.root.after(0, self._process_line, line)
                else:
                    time.sleep(0.01)
            except Exception as e:
                if self.serial_running:
                    self.root.after(0, self._log, f"Serial error: {e}\n")
                break

    def _send_cmd(self):
        cmd = self.send_var.get().strip()
        if cmd and self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write((cmd + "\n").encode())
            self._log(f">>> {cmd}\n")
            self.send_var.set("")

    def _quick_send(self, cmd):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write((cmd + "\n").encode())
            self._log(f">>> {cmd}\n")

    def _log(self, text):
        self.serial_text.config(state=tk.NORMAL)
        self.serial_text.insert(tk.END, text)
        self.serial_text.see(tk.END)
        if int(self.serial_text.index('end-1c').split('.')[0]) > 1500:
            self.serial_text.delete('1.0', '500.0')
        self.serial_text.config(state=tk.DISABLED)

    # ==================================================================
    #                       LINE PARSING
    # ==================================================================

    def _parse_csv_line(self, line):
        """Parse one CSV data line matching the DWM1000 initiator format.
        Returns a dict or None."""
        if not line or line.startswith('#') or line.startswith('='):
            return None
        if 'sample' in line.lower() and 'millis' in line.lower():
            return None  # header row
        parts = line.split(',')
        if len(parts) < 11:
            return None
        try:
            int(parts[0])  # sample number must be numeric
            return {
                "sample":            int(parts[0]),
                "millis":            int(parts[1]),
                "distance_m":        float(parts[2]),
                "round_trip_ticks":  int(parts[3]),
                "reply_delay_ticks": int(parts[4]),
                "tof_ticks":         int(parts[5]),
                "rx_power_dBm":      float(parts[6]),
                "fp_power_dBm":      float(parts[7]),
                "quality":           float(parts[8]),
                "temp_C":            float(parts[9]),
                "pressure_hPa":      float(parts[10]),
            }
        except (ValueError, IndexError):
            return None

    def _process_line(self, line):
        """Handle incoming serial line."""
        self._log(line + "\n")
        parsed = self._parse_csv_line(line)
        if parsed is None:
            return

        # ── If collecting, store the sample ──
        if self.collecting:
            true_d = float(self.true_dist_var.get())
            parsed["true_distance_m"] = true_d
            parsed["angle_deg"] = float(self.angle_var.get())
            parsed["antenna_delay"] = int(self.ant_delay_var.get())
            parsed["notes"] = self.notes_var.get()
            parsed["timestamp"] = datetime.now().isoformat()
            parsed["power_diff_dB"] = (parsed["rx_power_dBm"]
                                       - parsed["fp_power_dBm"])
            parsed["channel_condition"] = classify_channel(
                parsed["power_diff_dB"])
            parsed["confidence"] = compute_confidence(
                parsed["quality"], parsed["power_diff_dB"])
            parsed["error_m"] = parsed["distance_m"] - true_d

            self.session_buffer.append(parsed)
            self.sample_count += 1
            self.live_readings.append(parsed["distance_m"])
            if len(self.live_readings) > 200:
                self.live_readings = self.live_readings[-200:]

            arr = np.array(self.live_readings)
            self.sample_lbl.config(text=f"Samples: {self.sample_count}")
            self.stats_lbl.config(
                text=f"Mean: {arr.mean():.4f} m  |  "
                     f"Std: {arr.std():.4f} m  |  "
                     f"Error: {arr.mean()-true_d:+.4f} m")

        # ── If live correction active ──
        if self.live_correcting and self.model is not None:
            self._update_live(parsed)

    # ==================================================================
    #                      COLLECTION CONTROLS
    # ==================================================================

    def _start_collection(self):
        if not self.serial_conn or not self.serial_conn.is_open:
            messagebox.showwarning("Not Connected",
                                   "Connect to serial port first.")
            return
        try:
            float(self.true_dist_var.get())
        except ValueError:
            messagebox.showwarning("Invalid", "Enter a valid true distance.")
            return

        self.collecting = True
        self.session_buffer = []
        self.sample_count = 0
        self.live_readings = []
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.sample_lbl.config(text="Samples: 0")
        self.stats_lbl.config(text="Collecting...")

        # Optionally send Arduino commands
        if self.auto_start_var.get():
            dist = self.true_dist_var.get()
            self.serial_conn.write(f"distance {dist}\n".encode())
            time.sleep(0.1)
            self.serial_conn.write(b"start\n")
            self._log(f">>> distance {dist}\n>>> start\n")

        self._log(
            f"\n{'='*60}\n"
            f"  COLLECTION: true={self.true_dist_var.get()}m  "
            f"angle={self.angle_var.get()}  "
            f"delay={self.ant_delay_var.get()}\n"
            f"{'='*60}\n")

    def _stop_collection(self):
        self.collecting = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        n = len(self.session_buffer)
        self._log(f"\n--- STOPPED: {n} samples ---\n")

        if n > 0:
            new_df = pd.DataFrame(self.session_buffer)
            self.dataset = pd.concat([self.dataset, new_df],
                                     ignore_index=True)
            self._save_dataset()

            d = new_df["distance_m"]
            t = new_df["true_distance_m"].iloc[0]
            self._log(
                f"  True: {t:.4f}m | Mean: {d.mean():.4f}m | "
                f"Std: {d.std():.4f}m | Error: {d.mean()-t:+.4f}m\n"
                f"  Dataset total: {len(self.dataset)} samples\n\n")

    # ==================================================================
    #                     DATASET MANAGEMENT
    # ==================================================================

    def _save_dataset(self):
        if not self.dataset.empty:
            self.dataset.to_csv(self.dataset_path, index=False)

    def _load_existing_dataset(self):
        if os.path.exists(self.dataset_path):
            try:
                self.dataset = pd.read_csv(self.dataset_path)
            except Exception:
                self.dataset = pd.DataFrame()

    def _refresh_dataset_view(self):
        for item in self.ds_tree.get_children():
            self.ds_tree.delete(item)
        if self.dataset.empty:
            self.ds_info.config(text="Dataset: 0 samples")
            self._set_summary("")
            return

        df = self.dataset
        show = df.tail(500)
        for _, r in show.iterrows():
            td = r.get("true_distance_m", 0)
            md = r.get("distance_m", 0)
            self.ds_tree.insert("", tk.END, values=(
                f"{td:.4f}", f"{md:.4f}", f"{md-td:+.4f}",
                int(r.get("tof_ticks",0)),
                f"{r.get('rx_power_dBm',0):.1f}",
                f"{r.get('fp_power_dBm',0):.1f}",
                f"{r.get('quality',0):.1f}",
                f"{r.get('angle_deg',0):.0f}",
                f"{r.get('temp_C',0):.1f}",
                f"{r.get('pressure_hPa',0):.1f}"))

        self.ds_info.config(
            text=f"Dataset: {len(df)} samples (showing last {len(show)})")

        # Build summary
        if "true_distance_m" not in df.columns:
            return
        lines = [
            f"{'Config':<28}{'N':>6}  {'Mean Meas':>10}  "
            f"{'Mean Err':>10}  {'Std':>9}  {'MAE':>9}  "
            f"{'Mean RxPwr':>10}  {'Mean Qual':>10}",
            "-" * 100
        ]
        gcols = ["true_distance_m"]
        if "angle_deg" in df.columns and df["angle_deg"].nunique() > 1:
            gcols.append("angle_deg")

        for name, grp in df.groupby(gcols):
            if isinstance(name, tuple):
                lbl = f"{name[0]:.2f}m @ {name[1]:.0f} deg"
            else:
                lbl = f"{name:.2f}m"
            err = grp["distance_m"] - grp["true_distance_m"]
            lines.append(
                f"  {lbl:<26}{len(grp):>6}  {grp['distance_m'].mean():>10.4f}  "
                f"{err.mean():>+10.4f}  {err.std():>9.4f}  "
                f"{err.abs().mean():>9.4f}  "
                f"{grp['rx_power_dBm'].mean():>10.2f}  "
                f"{grp['quality'].mean():>10.2f}")

        lines.append("-" * 100)
        te = df["distance_m"] - df["true_distance_m"]
        lines.append(
            f"  {'TOTAL':<26}{len(df):>6}  {df['distance_m'].mean():>10.4f}  "
            f"{te.mean():>+10.4f}  {te.std():>9.4f}  "
            f"{te.abs().mean():>9.4f}  "
            f"{df['rx_power_dBm'].mean():>10.2f}  "
            f"{df['quality'].mean():>10.2f}")

        self._set_summary("\n".join(lines))

    def _set_summary(self, text):
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", text)
        self.summary_text.config(state=tk.DISABLED)

    def _import_file(self):
        """Import .csv or .txt calibration files."""
        paths = filedialog.askopenfilenames(
            filetypes=[("CSV/TXT", "*.csv *.txt"), ("All", "*.*")])
        if not paths:
            return

        total = 0
        for p in paths:
            try:
                recs = self._parse_cal_file(p)
                if recs:
                    self.dataset = pd.concat(
                        [self.dataset, pd.DataFrame(recs)],
                        ignore_index=True)
                    total += len(recs)
            except Exception as e:
                messagebox.showerror("Import Error",
                    f"{os.path.basename(p)}:\n{e}")

        if total > 0:
            self._save_dataset()
            self._refresh_dataset_view()
            messagebox.showinfo("Imported",
                f"Added {total} samples from {len(paths)} file(s).\n"
                f"Dataset total: {len(self.dataset)}")

    def _parse_cal_file(self, filepath):
        """Parse a DWM1000 calibration .txt or .csv file.
        Extracts metadata from comment headers."""
        records = []
        known_dist = None
        ant_delay = 0

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()

                # -- Extract metadata from comments / headers --
                if line.startswith('#') or line.startswith(' '):
                    low = line.lower()
                    m = re.search(
                        r'known\s+distance[^:]*:\s*([0-9]+\.?[0-9]*)', low)
                    if m:
                        known_dist = float(m.group(1))
                    m = re.search(
                        r'antenna\s+delay\s+register[^:]*:\s*([0-9]+)', low)
                    if m:
                        ant_delay = int(m.group(1))
                    continue

                # "Known distance set to: X.XXXX m" from serial log
                m = re.search(
                    r'Known distance set to:\s*([0-9]+\.?[0-9]*)', line)
                if m:
                    known_dist = float(m.group(1))
                    continue

                # -- Try parsing as CSV data --
                parsed = self._parse_csv_line(line)
                if parsed is not None and known_dist is not None:
                    parsed["true_distance_m"] = known_dist
                    parsed["antenna_delay"] = ant_delay
                    parsed["angle_deg"] = 0.0
                    parsed["notes"] = os.path.basename(filepath)
                    parsed["power_diff_dB"] = (parsed["rx_power_dBm"]
                                               - parsed["fp_power_dBm"])
                    parsed["channel_condition"] = classify_channel(
                        parsed["power_diff_dB"])
                    parsed["confidence"] = compute_confidence(
                        parsed["quality"], parsed["power_diff_dB"])
                    parsed["error_m"] = (parsed["distance_m"] - known_dist)
                    records.append(parsed)

        # If we got data but no known_dist from headers, ask user
        if not records and known_dist is None:
            # Try parsing without known_dist requirement
            temp_records = []
            with open(filepath, 'r') as f:
                for line in f:
                    parsed = self._parse_csv_line(line.strip())
                    if parsed:
                        temp_records.append(parsed)

            if temp_records:
                dist = simpledialog.askfloat(
                    "True Distance",
                    f"No known distance found in:\n"
                    f"{os.path.basename(filepath)}\n\n"
                    f"Enter the true distance (m):",
                    parent=self.root)
                if dist:
                    for r in temp_records:
                        r["true_distance_m"] = dist
                        r["antenna_delay"] = 0
                        r["angle_deg"] = 0.0
                        r["notes"] = os.path.basename(filepath)
                        r["power_diff_dB"] = (r["rx_power_dBm"]
                                              - r["fp_power_dBm"])
                        r["channel_condition"] = classify_channel(
                            r["power_diff_dB"])
                        r["confidence"] = compute_confidence(
                            r["quality"], r["power_diff_dB"])
                        r["error_m"] = r["distance_m"] - dist
                    records = temp_records

        return records

    def _export_csv(self):
        if self.dataset.empty:
            messagebox.showinfo("Empty", "No data to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.dataset.to_csv(path, index=False)
            messagebox.showinfo("Exported",
                f"{len(self.dataset)} samples saved to {path}")

    def _clear_dataset(self):
        if messagebox.askyesno("Clear", "Delete ALL collected data?"):
            self.dataset = pd.DataFrame()
            if os.path.exists(self.dataset_path):
                os.remove(self.dataset_path)
            self._refresh_dataset_view()

    # ==================================================================
    #                       ML TRAINING
    # ==================================================================

    def _get_features(self):
        return [k for k, v in self.feat_vars.items() if v.get()]

    def _build_X(self, df, features):
        """Build feature matrix, creating engineered columns as needed."""
        df = df.copy()
        if "power_diff_dB" in features and "power_diff_dB" not in df.columns:
            df["power_diff_dB"] = df["rx_power_dBm"] - df["fp_power_dBm"]
        if "channel_condition" in features and "channel_condition" not in df.columns:
            if "power_diff_dB" not in df.columns:
                df["power_diff_dB"] = df["rx_power_dBm"] - df["fp_power_dBm"]
            df["channel_condition"] = classify_channel(df["power_diff_dB"].values)
        if "confidence" in features and "confidence" not in df.columns:
            if "power_diff_dB" not in df.columns:
                df["power_diff_dB"] = df["rx_power_dBm"] - df["fp_power_dBm"]
            df["confidence"] = compute_confidence(
                df["quality"].values, df["power_diff_dB"].values)
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        X = df[features].values.astype(float)
        # Replace non-finite values with column medians
        for col_i in range(X.shape[1]):
            bad = ~np.isfinite(X[:, col_i])
            if bad.any():
                med = np.nanmedian(X[~bad, col_i]) if (~bad).any() else 0.0
                X[bad, col_i] = med
        return X

    def _train_model(self):
        if self.dataset.empty or len(self.dataset) < 10:
            messagebox.showwarning("Insufficient Data",
                f"Need >= 10 samples. Have {len(self.dataset)}.")
            return

        df = self.dataset.copy()
        features = self._get_features()
        if not features:
            messagebox.showwarning("No Features", "Select at least one.")
            return

        # Ensure engineered columns exist
        if "power_diff_dB" not in df.columns:
            df["power_diff_dB"] = df["rx_power_dBm"] - df["fp_power_dBm"]
        if "channel_condition" not in df.columns:
            df["channel_condition"] = classify_channel(df["power_diff_dB"].values)
        if "confidence" not in df.columns:
            df["confidence"] = compute_confidence(
                df["quality"].values, df["power_diff_dB"].values)

        try:
            X = self._build_X(df, features)
        except ValueError as e:
            messagebox.showerror("Feature Error", str(e))
            return

        y = df["true_distance_m"].values
        split = float(self.split_var.get())
        algo = self.algo_var.get()
        pdeg = int(self.poly_var.get())
        use_multistage = self.multistage_var.get()
        use_lodo = self.lodo_var.get()

        # ── Train/test split ──
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=split, random_state=42)

        # Build distance correction pipeline
        if algo == "polynomial_regression":
            model = Pipeline([
                ("poly", PolynomialFeatures(degree=pdeg, include_bias=False)),
                ("scaler", StandardScaler()),
                ("reg", LinearRegression())])
        elif algo == "ridge_polynomial":
            model = Pipeline([
                ("poly", PolynomialFeatures(degree=pdeg, include_bias=False)),
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0))])
        elif algo == "gradient_boosting":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", GradientBoostingRegressor(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42))])
        elif algo == "random_forest":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", RandomForestRegressor(
                    n_estimators=300, max_depth=8, random_state=42))])

        model.fit(X_tr, y_tr)

        y_pred_tr = model.predict(X_tr)
        y_pred_te = model.predict(X_te)

        # Raw baseline: find distance_m column index
        di = features.index("distance_m") if "distance_m" in features else None
        raw_te = X_te[:, di] if di is not None else df["distance_m"].values[:len(y_te)]

        raw_err = raw_te - y_te
        raw_mae = np.mean(np.abs(raw_err))
        raw_rmse = np.sqrt(np.mean(raw_err**2))

        cor_err = y_pred_te - y_te
        cor_mae = mean_absolute_error(y_te, y_pred_te)
        cor_rmse = np.sqrt(mean_squared_error(y_te, y_pred_te))
        r2 = r2_score(y_te, y_pred_te)

        # ── Cross-validation ──
        if use_lodo and "true_distance_m" in df.columns:
            groups = df["true_distance_m"].values
            n_groups = len(np.unique(groups))
            if n_groups >= 2:
                logo = LeaveOneGroupOut()
                cv_scores = cross_val_score(
                    model, X, y, groups=groups, cv=logo,
                    scoring='neg_mean_absolute_error')
                cv_label = f"LODO ({n_groups} distances)"
            else:
                cv_k = min(5, max(2, len(X) // 10))
                cv_scores = cross_val_score(model, X, y, cv=cv_k,
                                            scoring='neg_mean_absolute_error')
                cv_label = f"{cv_k}-fold (only 1 distance, LODO not possible)"
        else:
            cv_k = min(5, max(2, len(X) // 10))
            cv_scores = cross_val_score(model, X, y, cv=cv_k,
                                        scoring='neg_mean_absolute_error')
            cv_label = f"{cv_k}-fold"

        # ── NLOS classification stage (Stage 1) ──
        nlos_txt = ""
        nlos_model = None
        if use_multistage and "power_diff_dB" in df.columns:
            ch_labels = classify_channel(df["power_diff_dB"].values).astype(int)
            n_classes = len(np.unique(ch_labels))
            ch_names = {0: "LOS", 1: "Marginal", 2: "NLOS"}

            # Distribution summary
            nlos_txt = "\n--- STAGE 1: CHANNEL CLASSIFICATION ---\n"
            nlos_txt += "  Threshold-based classification (DW1000 manual Sec 4.7):\n"
            for c in sorted(np.unique(ch_labels)):
                count = (ch_labels == c).sum()
                pct = count / len(ch_labels) * 100
                nlos_txt += f"    {ch_names.get(c, '?'):<12} (class {c}): "
                nlos_txt += f"{count:>5} samples ({pct:.1f}%)\n"

            # Train a classifier if we have mixed conditions
            if n_classes >= 2:
                nlos_feats = ["rx_power_dBm", "fp_power_dBm", "quality",
                              "power_diff_dB"]
                nlos_feats = [f for f in nlos_feats if f in df.columns]
                if nlos_feats:
                    X_nlos = self._build_X(df, nlos_feats)
                    X_n_tr, X_n_te, y_n_tr, y_n_te = train_test_split(
                        X_nlos, ch_labels, test_size=0.2, random_state=42)
                    nlos_clf = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", RandomForestClassifier(
                            n_estimators=200, max_depth=6, random_state=42))])
                    nlos_clf.fit(X_n_tr, y_n_tr)
                    y_n_pred = nlos_clf.predict(X_n_te)
                    n_acc = accuracy_score(y_n_te, y_n_pred)
                    nlos_model = nlos_clf

                    nlos_txt += f"\n  Trained RF classifier on: {', '.join(nlos_feats)}\n"
                    nlos_txt += f"  Accuracy: {n_acc*100:.1f}%\n"

                    # Per-class correction performance
                    nlos_txt += "\n  Per-channel-condition distance error:\n"
                    for c in sorted(np.unique(ch_labels)):
                        mask = ch_labels == c
                        if mask.sum() == 0:
                            continue
                        sX = self._build_X(df[mask], features)
                        sy = df.loc[mask, "true_distance_m"].values
                        sp = model.predict(sX)
                        cm = np.mean(np.abs(sp - sy))
                        if di is not None:
                            sr = sX[:, di]
                            rm = np.mean(np.abs(sr - sy))
                            nlos_txt += (f"    {ch_names.get(c, '?'):<12}: "
                                         f"Raw MAE={rm*100:.1f}cm -> "
                                         f"Corrected MAE={cm*100:.1f}cm  "
                                         f"({mask.sum()} samples)\n")
                        else:
                            nlos_txt += (f"    {ch_names.get(c, '?'):<12}: "
                                         f"Corrected MAE={cm*100:.1f}cm  "
                                         f"({mask.sum()} samples)\n")
            else:
                nlos_txt += f"\n  All samples are same class, "
                nlos_txt += "classifier not trained.\n"

            # Confidence distribution
            if "confidence" in df.columns:
                conf = df["confidence"].values
                nlos_txt += f"\n  Confidence: mean={conf.mean():.3f}  "
                nlos_txt += f"min={conf.min():.3f}  max={conf.max():.3f}\n"
                low_conf = (conf < 0.3).sum()
                if low_conf > 0:
                    nlos_txt += (f"  Warning: {low_conf} samples ({low_conf/len(conf)*100:.1f}%) "
                                 f"have low confidence (< 0.3)\n")

        # Feature importance for tree models
        imp_str = ""
        if algo in ("gradient_boosting", "random_forest"):
            imp = model.named_steps["reg"].feature_importances_
            idx = np.argsort(imp)[::-1]
            imp_str = "\n--- STAGE 2: FEATURE IMPORTANCE ---\n"
            for i in idx:
                bar = "#" * int(imp[i] * 40)
                imp_str += f"  {features[i]:<22} {imp[i]:.4f}  {bar}\n"

        self.model = model
        self.model_features = features
        self.nlos_model = nlos_model if use_multistage else None
        improve = ((raw_mae - cor_mae) / raw_mae * 100) if raw_mae > 0 else 0
        self.model_info = {
            "algorithm": algo, "features": features,
            "raw_mae": raw_mae, "cor_mae": cor_mae,
            "improvement": improve,
            "multistage": use_multistage,
            "nlos_model": nlos_model,
        }

        # ── Results text ──
        pipeline_label = "MULTI-STAGE" if use_multistage else "SINGLE-STAGE"
        txt = (
            f"{'='*55}\n"
            f"  ML MODEL TRAINING RESULTS ({pipeline_label})\n"
            f"{'='*55}\n\n"
            f"Algorithm:  {algo}\n"
            f"Features:   {', '.join(features)}\n"
            f"Train/Test: {len(X_tr)} / {len(X_te)}\n"
            f"Pipeline:   {pipeline_label}\n\n"
            f"{'_'*40}\n"
            f"  BEFORE ML (raw vs true):\n"
            f"    MAE:  {raw_mae:.4f} m  ({raw_mae*100:.2f} cm)\n"
            f"    RMSE: {raw_rmse:.4f} m  ({raw_rmse*100:.2f} cm)\n"
            f"    Mean bias: {raw_err.mean():+.4f} m\n\n"
            f"  AFTER ML Correction:\n"
            f"    MAE:  {cor_mae:.4f} m  ({cor_mae*100:.2f} cm)\n"
            f"    RMSE: {cor_rmse:.4f} m  ({cor_rmse*100:.2f} cm)\n"
            f"    R2:   {r2:.6f}\n"
            f"    Mean bias: {cor_err.mean():+.4f} m\n\n"
            f"  IMPROVEMENT: {improve:.1f}% MAE reduction\n"
            f"{'_'*40}\n"
            f"  Cross-Validation ({cv_label}):\n"
            f"    MAE: {-cv_scores.mean():.4f} +/- {cv_scores.std():.4f} m\n"
            f"{nlos_txt}"
            f"{imp_str}\n")

        # Per-distance breakdown
        if df["true_distance_m"].nunique() > 1:
            txt += "\n--- PER-DISTANCE BREAKDOWN ---\n"
            for dist in sorted(df["true_distance_m"].unique()):
                mask = df["true_distance_m"] == dist
                sX = self._build_X(df[mask], features)
                sy = df.loc[mask, "true_distance_m"].values
                sp = model.predict(sX)
                if di is not None:
                    sr = sX[:, di]
                    rm = np.mean(np.abs(sr - sy))
                else:
                    rm = float('nan')
                cm = np.mean(np.abs(sp - sy))
                txt += (f"  {dist:.2f}m: Raw MAE={rm:.4f}m -> "
                        f"Corrected MAE={cm:.4f}m  ({len(sy)} samples)\n")

        # LODO per-fold results
        if use_lodo and "true_distance_m" in df.columns:
            groups = df["true_distance_m"].values
            n_groups = len(np.unique(groups))
            if n_groups >= 2:
                txt += "\n--- LODO FOLD DETAILS ---\n"
                logo = LeaveOneGroupOut()
                for fold_i, (tr_idx, te_idx) in enumerate(logo.split(X, y, groups)):
                    held_dist = groups[te_idx[0]]
                    fold_model = type(model).from_pipeline(model) if hasattr(type(model), 'from_pipeline') else None
                    # Just report the CV score per fold
                    txt += f"  Fold {fold_i+1}: held out {held_dist:.2f}m "
                    txt += f"({len(te_idx)} samples)  "
                    txt += f"MAE = {-cv_scores[fold_i]*100:.1f} cm\n"

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", txt)
        self.results_text.config(state=tk.DISABLED)

        self.model_status.config(
            text=f"Trained: {algo} (MAE: {cor_mae:.4f}m) [{pipeline_label}]",
            foreground="green")

        # ── Plot ──
        self._plot_results(X_te, y_te, y_pred_te, raw_te, df, features, di)

    def _plot_results(self, X_te, y_te, y_pred, raw_te, df, features, di):
        self.fig.clear()
        ax1 = self.fig.add_subplot(231)
        ax2 = self.fig.add_subplot(232)
        ax3 = self.fig.add_subplot(233)
        ax4 = self.fig.add_subplot(234)
        ax5 = self.fig.add_subplot(235)
        ax6 = self.fig.add_subplot(236)

        re = raw_te - y_te
        ce = y_pred - y_te

        # 1 - Raw vs Corrected scatter
        ax1.scatter(y_te, raw_te, alpha=0.3, s=10, c="red", label="Raw")
        ax1.scatter(y_te, y_pred, alpha=0.3, s=10, c="green", label="Corrected")
        lo = min(y_te.min(), raw_te.min(), y_pred.min()) - 0.3
        hi = max(y_te.max(), raw_te.max(), y_pred.max()) + 0.3
        ax1.plot([lo,hi], [lo,hi], 'k--', alpha=0.5, label="Perfect")
        ax1.set_xlabel("True (m)"); ax1.set_ylabel("Output (m)")
        ax1.set_title("Raw vs Corrected", fontsize=9); ax1.legend(fontsize=6)

        # 2 - Error histograms
        ax2.hist(re, bins=40, alpha=0.5, color="red",
                 label=f"Raw s={re.std():.4f}")
        ax2.hist(ce, bins=40, alpha=0.5, color="green",
                 label=f"Corr s={ce.std():.4f}")
        ax2.axvline(0, color='k', ls='--', alpha=0.5)
        ax2.set_xlabel("Error (m)"); ax2.set_title("Error Distribution", fontsize=9)
        ax2.legend(fontsize=6)

        # 3 - Error vs true distance (full dataset)
        aX = self._build_X(df, features)
        ay = df["true_distance_m"].values
        ap = self.model.predict(aX)
        if di is not None:
            ar = aX[:, di]
        else:
            ar = df["distance_m"].values
        ax3.scatter(ay, ar-ay, alpha=0.2, s=6, c="red", label="Raw")
        ax3.scatter(ay, ap-ay, alpha=0.2, s=6, c="green", label="Corrected")
        ax3.axhline(0, color='k', ls='--', alpha=0.5)
        ax3.set_xlabel("True (m)"); ax3.set_ylabel("Error (m)")
        ax3.set_title("Error vs Distance", fontsize=9); ax3.legend(fontsize=6)

        # 4 - Residual plot
        ax4.scatter(ap, ap-ay, alpha=0.2, s=6, c="green")
        ax4.axhline(0, color='k', ls='--', alpha=0.5)
        ax4.set_xlabel("Predicted (m)"); ax4.set_ylabel("Residual (m)")
        ax4.set_title("Residual Plot", fontsize=9)

        # 5 - Channel condition distribution
        if "power_diff_dB" in df.columns:
            pd_vals = df["power_diff_dB"].values
            ch_labels = classify_channel(pd_vals)
            colors_map = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
            ch_names = {0: "LOS", 1: "Marginal", 2: "NLOS"}
            for c in [0, 1, 2]:
                mask = ch_labels == c
                if mask.any():
                    ax5.scatter(
                        ay[mask], (ar-ay)[mask] if di is not None else np.zeros(mask.sum()),
                        alpha=0.4, s=8, c=colors_map[c],
                        label=f"{ch_names[c]} ({mask.sum()})")
            ax5.axhline(0, color='k', ls='--', alpha=0.5)
            ax5.axhspan(-0.1, 0.1, alpha=0.1, color='green')
            ax5.set_xlabel("True (m)"); ax5.set_ylabel("Raw Error (m)")
            ax5.set_title("Error by Channel Condition", fontsize=9)
            ax5.legend(fontsize=6)
        else:
            ax5.text(0.5, 0.5, "No power_diff_dB data",
                     ha='center', va='center', transform=ax5.transAxes)

        # 6 - Confidence vs absolute error
        if "confidence" in df.columns or "power_diff_dB" in df.columns:
            if "confidence" not in df.columns:
                conf = compute_confidence(
                    df["quality"].values, df["power_diff_dB"].values)
            else:
                conf = df["confidence"].values
            abs_err = np.abs(ap - ay)
            ax6.scatter(conf, abs_err, alpha=0.2, s=6, c="blue")
            # Add trend line
            if len(conf) > 10:
                bins = np.linspace(0, 1, 11)
                bin_idx = np.digitize(conf, bins)
                for b in range(1, len(bins)):
                    mask = bin_idx == b
                    if mask.sum() > 0:
                        ax6.plot(bins[b-1:b+1].mean(), abs_err[mask].mean(),
                                 'ro', ms=6, zorder=5)
            ax6.set_xlabel("Confidence"); ax6.set_ylabel("|Error| (m)")
            ax6.set_title("Confidence vs Error", fontsize=9)
        else:
            ax6.text(0.5, 0.5, "No confidence data",
                     ha='center', va='center', transform=ax6.transAxes)

        self.fig.tight_layout()
        self.canvas.draw()

    def _save_model(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl", filetypes=[("Pickle", "*.pkl")])
        if path:
            with open(path, 'wb') as f:
                pickle.dump({"model": self.model,
                             "features": self.model_features,
                             "info": self.model_info,
                             "nlos_model": getattr(self, 'nlos_model', None)}, f)
            messagebox.showinfo("Saved", f"Model saved to {path}")

    def _load_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("Pickle", "*.pkl")])
        if path:
            try:
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                self.model = obj["model"]
                self.model_features = obj["features"]
                self.model_info = obj.get("info", {})
                self.nlos_model = obj.get("nlos_model", None)
                ms = "MULTI-STAGE" if self.model_info.get("multistage") else "SINGLE"
                self.model_status.config(
                    text=f"Loaded: {self.model_info.get('algorithm','?')} [{ms}]",
                    foreground="green")
            except Exception as e:
                messagebox.showerror("Load Error", str(e))

    # ==================================================================
    #                     LIVE CORRECTION
    # ==================================================================

    def _toggle_live(self):
        if self.live_correcting:
            self.live_correcting = False
            self.live_btn.config(text="Start Live Correction")
        else:
            if self.model is None:
                messagebox.showwarning("No Model",
                                       "Train or load a model first.")
                return
            if not self.serial_conn or not self.serial_conn.is_open:
                messagebox.showwarning("Not Connected",
                                       "Connect serial first.")
                return
            self.live_correcting = True
            self.live_raw_history = []
            self.live_corr_history = []
            self.live_btn.config(text="Stop Live Correction")

    def _update_live(self, parsed):
        """Apply multi-stage ML correction to a parsed line in real-time."""
        try:
            true_d = float(self.live_true_var.get())
        except ValueError:
            return

        p = dict(parsed)
        if "power_diff_dB" not in p:
            p["power_diff_dB"] = p["rx_power_dBm"] - p["fp_power_dBm"]
        if "angle_deg" not in p:
            p["angle_deg"] = float(self.live_angle_var.get())
        if "channel_condition" not in p:
            p["channel_condition"] = classify_channel(p["power_diff_dB"])
        if "confidence" not in p:
            p["confidence"] = compute_confidence(
                p["quality"], p["power_diff_dB"])

        try:
            fv = [p[f] for f in self.model_features]
        except KeyError:
            return

        corrected = self.model.predict(np.array([fv]))[0]
        raw_d = p["distance_m"]
        raw_e = raw_d - true_d
        cor_e = corrected - true_d

        # Channel condition label
        ch_names = {0: "LOS", 1: "MARGINAL", 2: "NLOS"}
        ch_colors = {0: "green", 1: "orange", 2: "red"}
        ch = int(p["channel_condition"])
        conf = p["confidence"]

        self.raw_val.config(text=f"{raw_d:.4f} m")
        self.raw_err.config(
            text=f"Error: {raw_e:+.4f} m  |  "
                 f"{ch_names.get(ch, '?')}  conf={conf:.2f}",
            foreground=ch_colors.get(ch, "gray"))
        self.cor_val.config(text=f"{corrected:.4f} m")
        self.cor_err.config(
            text=f"Error: {cor_e:+.4f} m",
            foreground="green" if abs(cor_e) < abs(raw_e) else "orange")

        self.live_raw_history.append(raw_d)
        self.live_corr_history.append(corrected)
        if len(self.live_raw_history) > 200:
            self.live_raw_history = self.live_raw_history[-200:]
            self.live_corr_history = self.live_corr_history[-200:]

        w = 50
        if len(self.live_raw_history) >= 5:
            rw = np.array(self.live_raw_history[-w:])
            cw = np.array(self.live_corr_history[-w:])
            self.roll_raw.config(
                text=f"Raw  u: {rw.mean():.4f}  s: {rw.std():.4f}")
            self.roll_cor.config(
                text=f"Corr u: {cw.mean():.4f}  s: {cw.std():.4f}")
            self.roll_mae.config(
                text=f"Raw MAE: {np.mean(np.abs(rw-true_d)):.4f}  "
                     f"Corr MAE: {np.mean(np.abs(cw-true_d)):.4f}")

        if len(self.live_raw_history) % 5 == 0:
            self._update_live_plot(true_d)

    def _update_live_plot(self, true_d):
        self.live_fig.clear()
        ax = self.live_fig.add_subplot(111)
        x = range(len(self.live_raw_history))
        ax.plot(x, self.live_raw_history, 'r-', alpha=0.5, lw=0.8,
                label="Raw")
        ax.plot(x, self.live_corr_history, 'g-', alpha=0.8, lw=1.0,
                label="Corrected")
        ax.axhline(true_d, color='blue', ls='--', alpha=0.7,
                   label=f"True ({true_d:.3f}m)")
        ax.set_xlabel("Sample"); ax.set_ylabel("Distance (m)")
        ax.legend(fontsize=8, loc="upper right")
        self.live_fig.tight_layout()
        self.live_canvas.draw()

    # ==================================================================
    #                         CLEANUP
    # ==================================================================

    def on_close(self):
        self.serial_running = False
        self.collecting = False
        self.live_correcting = False
        if self.serial_conn:
            try: self.serial_conn.close()
            except: pass
        self.root.destroy()


# ==================================================================
def main():
    root = tk.Tk()
    app = UWBCalibrationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()