#!/usr/bin/env python3
"""
UWB BLE Dashboard — Raspberry Pi 5
Connects to all T* (tag) and A* (anchor) devices over BLE,
collects UWB ranging + signal-quality data, logs to CSV,
and serves a live web dashboard on http://<pi-ip>:5000

Usage:
    pip install bleak flask --break-system-packages
    python3 uwb_dashboard.py

Press Ctrl+C to stop.  Data is saved in logs/ directory.
"""

import asyncio
import struct
import threading
import signal
import csv
import math
import datetime
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from bleak import BleakScanner, BleakClient
from flask import Flask, render_template, jsonify, make_response, request

# ─────────────────────── CONFIG ───────────────────────
TAG_CHAR_UUID    = "19b10011-e8f2-537e-4f6c-d104768a1214"
ANCHOR_CHAR_UUID = "19b10012-e8f2-537e-4f6c-d104768a1214"
CMD_CHAR_UUID    = "19b10013-e8f2-537e-4f6c-d104768a1214"

# Device names to look for (expand as needed)
TAG_NAMES    = [f"T{i}" for i in range(1, 11)]
ANCHOR_NAMES = [f"A{i}" for i in range(1, 11)]
ALL_NAMES    = TAG_NAMES + ANCHOR_NAMES

SCAN_TIMEOUT    = 8.0    # BLE scan duration per sweep (seconds)
RESCAN_INTERVAL = 15.0   # seconds between sweeps (picks up late-joining devices)
RECONNECT_SEC   = 3      # seconds to wait before each reconnect attempt
HISTORY_LEN     = 200    # packets kept in memory per device
FLASK_PORT      = 5050

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Current antenna delay – used to route logs into AD_<value> subfolders
current_antenna_delay = 0   # 0 = unknown / not yet set

# ─────────────────────── SHARED STATE ───────────────────────
lock = threading.Lock()

# device_name → { "type": "tag"|"anchor", "connected": bool,
#                  "latest": dict, "history": deque, "addr": str,
#                  "connect_time": str, "packet_count": int,
#                  "settings": dict }
devices = {}

# Active BleakClient references for sending commands
active_clients = {}   # device_name → BleakClient

# Command queue: Flask thread appends; BLE thread processes
pending_commands = []  # list of (device_name, command_string)
cmd_lock = threading.Lock()

session_start = datetime.datetime.now().isoformat(timespec="seconds")

# ─────────────────────── DATA COLLECTION STATE ───────────────────────
collection_lock = threading.Lock()
collection = {
    "active": False,
    "true_dist_m": 0.0,
    "angle_deg": 0.0,
    "notes": "",
    "session_id": "",
    "target_samples": 500,
    "samples": [],          # list of labelled dicts
}

def _ad_subdir():
    """Return the antenna-delay subfolder, e.g. logs/AD_16700/."""
    ad = current_antenna_delay
    name = f"AD_{ad}" if ad else "AD_unknown"
    d = LOG_DIR / name
    d.mkdir(exist_ok=True)
    return d

def _cal_dataset_path():
    """Return the per-antenna-delay calibration CSV path."""
    return _ad_subdir() / "ble_cal_dataset.csv"
CAL_COLUMNS = [
    "timestamp", "device", "session_id", "seq",
    "true_dist_m", "angle_deg", "distance_m",
    "rx_power", "fp_power", "fp_rx_ratio", "quality",
    "std_noise", "fp_ampl1", "fp_ampl2", "fp_ampl3",
    "cir_power", "rxpacc", "nlos_suspect", "anchor_id",
    "error_m", "antenna_delay", "notes",
]


def _collect_tag_packet(name, pkt):
    """If collection is active, label and store the tag packet."""
    with collection_lock:
        if not collection["active"]:
            return
        d = pkt.get("distance_m", float("nan"))
        td = collection["true_dist_m"]
        err = d - td if not math.isnan(d) else float("nan")
        row = {
            "timestamp":    pkt.get("_ts", datetime.datetime.now().isoformat(timespec="milliseconds")),
            "device":       name,
            "session_id":   collection["session_id"],
            "seq":          pkt.get("seq", 0),
            "true_dist_m":  td,
            "angle_deg":    collection["angle_deg"],
            "distance_m":   d,
            "rx_power":     pkt.get("rx_power", float("nan")),
            "fp_power":     pkt.get("fp_power", float("nan")),
            "fp_rx_ratio":  pkt.get("fp_rx_ratio", float("nan")),
            "quality":      pkt.get("quality", float("nan")),
            "std_noise":    pkt.get("std_noise", float("nan")),
            "fp_ampl1":     pkt.get("fp_ampl1", float("nan")),
            "fp_ampl2":     pkt.get("fp_ampl2", float("nan")),
            "fp_ampl3":     pkt.get("fp_ampl3", float("nan")),
            "cir_power":    pkt.get("cir_power", float("nan")),
            "rxpacc":       pkt.get("rxpacc", float("nan")),
            "nlos_suspect": pkt.get("nlos_suspect", False),
            "anchor_id":    pkt.get("anchor_id", 0),
            "error_m":      err,
            "antenna_delay": current_antenna_delay,
            "notes":        collection["notes"],
        }
        collection["samples"].append(row)
        # Auto-stop if target reached
        if len(collection["samples"]) >= collection["target_samples"]:
            collection["active"] = False


def _save_collection():
    """Append collected samples to the cal dataset CSV."""
    with collection_lock:
        samples = list(collection["samples"])
    if not samples:
        return 0
    new_df = pd.DataFrame(samples, columns=CAL_COLUMNS)
    cal_path = _cal_dataset_path()
    if cal_path.exists():
        existing = pd.read_csv(cal_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(cal_path, index=False)
    return len(samples)


def init_device(name, addr, dev_type):
    """Register a device in shared state."""
    with lock:
        if name not in devices:
            devices[name] = {
                "type": dev_type,
                "connected": False,
                "addr": addr,
                "connect_time": None,
                "latest": {},
                "history": deque(maxlen=HISTORY_LEN),
                "packet_count": 0,
                "settings": {"antenna_delay": 0, "range_interval": 500},
            }


# ─────────────────────── STRUCT UNPACKING ───────────────────────
# TagFrame (43 bytes):
#   anchor_id(B) seq(H) distance_m(f)
#   round_trip_lo(i) round_trip_hi(B)
#   reply_delay_lo(i) reply_delay_hi(B)
#   rx_power(f) fp_power(f) quality(f)
#   std_noise(H) fp_ampl1(H) fp_ampl2(H) fp_ampl3(H) cir_power(H) rxpacc(H)
#   flags(B) anchor_count(B)
TAG_FMT = "<BHfiBiBfffHHHHHHBB"
TAG_SIZE = struct.calcsize(TAG_FMT)   # 43

# AnchorFrame (33 bytes):
#   tag_id(B) seq(H)
#   rx_power(f) fp_power(f) quality(f)
#   std_noise(H) fp_ampl1(H) fp_ampl2(H) fp_ampl3(H) cir_power(H) rxpacc(H)
#   reply_delay_lo(i) reply_delay_hi(B) flags(B)
ANCHOR_FMT = "<BHfffHHHHHHiBB"
ANCHOR_SIZE = struct.calcsize(ANCHOR_FMT)   # 33


def unpack_tag(data: bytes) -> dict:
    if len(data) < TAG_SIZE:
        return None
    vals = struct.unpack(TAG_FMT, data[:TAG_SIZE])
    # [0]anchor_id [1]seq [2]distance_m [3]rt_lo [4]rt_hi
    # [5]rd_lo [6]rd_hi [7]rx_power [8]fp_power [9]quality
    # [10]std_noise [11]fp_ampl1 [12]fp_ampl2 [13]fp_ampl3
    # [14]cir_power [15]rxpacc [16]flags [17]anchor_count
    round_trip  = (vals[4] << 32) | (vals[3] & 0xFFFFFFFF)
    reply_delay = (vals[6] << 32) | (vals[5] & 0xFFFFFFFF)
    return {
        "anchor_id":    vals[0],
        "seq":          vals[1],
        "distance_m":   round(vals[2], 3),
        "round_trip":   round_trip,
        "reply_delay":  reply_delay,
        "rx_power":     round(vals[7], 1),
        "fp_power":     round(vals[8], 1),
        "fp_rx_ratio":  round(vals[8] - vals[7], 1),
        "quality":      round(vals[9], 2),
        "std_noise":    vals[10],
        "fp_ampl1":     vals[11],
        "fp_ampl2":     vals[12],
        "fp_ampl3":     vals[13],
        "cir_power":    vals[14],
        "rxpacc":       vals[15],
        "flags":        vals[16],
        "anchor_count": vals[17],
        "nlos_suspect": bool(vals[16] & 0x02),
    }


def unpack_anchor(data: bytes) -> dict:
    if len(data) < ANCHOR_SIZE:
        return None
    vals = struct.unpack(ANCHOR_FMT, data[:ANCHOR_SIZE])
    # [0]tag_id [1]seq [2]rx_power [3]fp_power [4]quality
    # [5]std_noise [6]fp_ampl1 [7]fp_ampl2 [8]fp_ampl3
    # [9]cir_power [10]rxpacc [11]rd_lo [12]rd_hi [13]flags
    reply_delay = (vals[12] << 32) | (vals[11] & 0xFFFFFFFF)
    return {
        "tag_id":       vals[0],
        "seq":          vals[1],
        "rx_power":     round(vals[2], 1),
        "fp_power":     round(vals[3], 1),
        "fp_rx_ratio":  round(vals[3] - vals[2], 1),
        "quality":      round(vals[4], 2),
        "std_noise":    vals[5],
        "fp_ampl1":     vals[6],
        "fp_ampl2":     vals[7],
        "fp_ampl3":     vals[8],
        "cir_power":    vals[9],
        "rxpacc":       vals[10],
        "reply_delay":  reply_delay,
        "flags":        vals[13],
    }


# ─────────────────────── CSV LOGGER ───────────────────────
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

def _open_log_csvs(label=""):
    """Open (or re-open) tag/anchor CSV log files in the current AD subfolder.

    If *label* is given (e.g. '1.0m_0deg') it becomes the leading part of
    the filename so you can identify the distance at a glance:
        tag_1.0m_0deg_20260326_143523.csv
    Without a label, generic logs are created:
        tag_20260326_143523.csv
    """
    global tag_csv, anchor_csv, tag_writer, anchor_writer, tag_csv_path, anchor_csv_path
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{label}_" if label else ""
    subdir = _ad_subdir()
    tag_csv_path    = subdir / f"tag_{prefix}{ts}.csv"
    anchor_csv_path = subdir / f"anchor_{prefix}{ts}.csv"
    tag_csv    = open(tag_csv_path, "w", newline="")
    anchor_csv = open(anchor_csv_path, "w", newline="")
    tag_writer = csv.writer(tag_csv)
    tag_writer.writerow(TAG_HEADER)
    anchor_writer = csv.writer(anchor_csv)
    anchor_writer.writerow(ANCHOR_HEADER)
    print(f"  Log dir: {subdir.resolve()}  ({tag_csv_path.name})")

_open_log_csvs()


def log_tag(name, pkt):
    now = datetime.datetime.now().isoformat(timespec="milliseconds")
    tag_writer.writerow([
        now, name, pkt["anchor_id"], pkt["seq"], pkt["distance_m"],
        pkt["rx_power"], pkt["fp_power"], pkt["fp_rx_ratio"],
        pkt["quality"], pkt["round_trip"], pkt["reply_delay"],
        pkt["std_noise"], pkt["fp_ampl1"], pkt["fp_ampl2"], pkt["fp_ampl3"],
        pkt["cir_power"], pkt["rxpacc"], pkt["flags"],
        pkt["anchor_count"], pkt["nlos_suspect"],
    ])
    tag_csv.flush()


def log_anchor(name, pkt):
    now = datetime.datetime.now().isoformat(timespec="milliseconds")
    anchor_writer.writerow([
        now, name, pkt["tag_id"], pkt["seq"],
        pkt["rx_power"], pkt["fp_power"], pkt["fp_rx_ratio"],
        pkt["quality"], pkt["std_noise"], pkt["fp_ampl1"],
        pkt["fp_ampl2"], pkt["fp_ampl3"], pkt["cir_power"],
        pkt["rxpacc"], pkt["reply_delay"], pkt["flags"],
    ])
    anchor_csv.flush()


# ─────────────────────── BLE HANDLERS ───────────────────────

async def _refresh_device(name: str, current):
    """
    After a disconnect, try a short BLE scan to get a fresh device object.
    This handles cases where the Arduino rebooted and got a different address,
    or where the stale Bleak object can no longer connect.
    Returns a fresh BleakDevice if found, otherwise returns current unchanged.
    """
    try:
        fresh = await BleakScanner.find_device_by_name(name, timeout=6.0)
        if fresh:
            if fresh.address != current.address:
                print(f"[{name}] Address changed {current.address} → {fresh.address}")
            return fresh
    except Exception as e:
        print(f"[{name}] Re-scan error: {e}")
    return current


async def handle_tag(ble_device):
    name    = ble_device.name
    current = ble_device
    init_device(name, current.address, "tag")

    while True:
        try:
            async with BleakClient(current, timeout=15.0) as client:
                with lock:
                    devices[name]["connected"]    = True
                    devices[name]["addr"]         = current.address
                    devices[name]["connect_time"] = (
                        datetime.datetime.now().isoformat(timespec="seconds")
                    )
                print(f"[+] Connected: {name} [{current.address}]")

                def on_notify(_, data):
                    pkt = unpack_tag(data)
                    if pkt is None:
                        return
                    pkt["_ts"] = datetime.datetime.now().isoformat(
                        timespec="milliseconds"
                    )
                    with lock:
                        devices[name]["latest"] = pkt
                        devices[name]["history"].append(pkt)
                        devices[name]["packet_count"] += 1
                    log_tag(name, pkt)
                    _collect_tag_packet(name, pkt)

                    flag = " NLOS?" if pkt["nlos_suspect"] else ""
                    print(
                        f"  {name}→A{pkt['anchor_id']} #{pkt['seq']:>5}  "
                        f"d={pkt['distance_m']:>7.3f}m  "
                        f"RX={pkt['rx_power']:>6.1f}  "
                        f"FP={pkt['fp_power']:>6.1f}  "
                        f"Q={pkt['quality']:>6.2f}  "
                        f"RT={pkt['round_trip']}  "
                        f"RD={pkt['reply_delay']}{flag}"
                    )

                await client.start_notify(TAG_CHAR_UUID, on_notify)

                # Store client reference for command sending
                active_clients[name] = client

                # Query initial settings
                try:
                    await client.write_gatt_char(CMD_CHAR_UUID, b"ST")
                    await asyncio.sleep(0.3)
                    resp = await client.read_gatt_char(CMD_CHAR_UUID)
                    settings_str = resp.decode("utf-8", errors="replace")
                    _parse_settings(name, settings_str)
                    print(f"[{name}] Settings: {settings_str}")
                except Exception as e:
                    print(f"[{name}] Could not read settings: {e}")

                while client.is_connected:
                    # Process any pending commands for this device
                    await _process_pending_commands(name, client)
                    await asyncio.sleep(0.5)

        except Exception as e:
            print(f"[!] {name}: {e}")

        active_clients.pop(name, None)
        with lock:
            devices[name]["connected"] = False
        print(f"[-] {name} disconnected — retry in {RECONNECT_SEC}s")
        await asyncio.sleep(RECONNECT_SEC)

        # Refresh device object in case the Arduino rebooted/changed address
        current = await _refresh_device(name, current)


async def handle_anchor(ble_device):
    name    = ble_device.name
    current = ble_device
    init_device(name, current.address, "anchor")

    while True:
        try:
            async with BleakClient(current, timeout=15.0) as client:
                with lock:
                    devices[name]["connected"]    = True
                    devices[name]["addr"]         = current.address
                    devices[name]["connect_time"] = (
                        datetime.datetime.now().isoformat(timespec="seconds")
                    )
                print(f"[+] Connected: {name} [{current.address}]")

                def on_notify(_, data):
                    pkt = unpack_anchor(data)
                    if pkt is None:
                        return
                    pkt["_ts"] = datetime.datetime.now().isoformat(
                        timespec="milliseconds"
                    )
                    with lock:
                        devices[name]["latest"] = pkt
                        devices[name]["history"].append(pkt)
                        devices[name]["packet_count"] += 1
                    log_anchor(name, pkt)

                    print(
                        f"  {name}←T{pkt['tag_id']} #{pkt['seq']:>5}  "
                        f"RX={pkt['rx_power']:>6.1f}  "
                        f"FP={pkt['fp_power']:>6.1f}  "
                        f"Q={pkt['quality']:>6.2f}  "
                        f"RD={pkt['reply_delay']}"
                    )

                await client.start_notify(ANCHOR_CHAR_UUID, on_notify)

                # Store client reference for command sending
                active_clients[name] = client

                # Query initial settings
                try:
                    await client.write_gatt_char(CMD_CHAR_UUID, b"ST")
                    await asyncio.sleep(0.3)
                    resp = await client.read_gatt_char(CMD_CHAR_UUID)
                    settings_str = resp.decode("utf-8", errors="replace")
                    _parse_settings(name, settings_str)
                    print(f"[{name}] Settings: {settings_str}")
                except Exception as e:
                    print(f"[{name}] Could not read settings: {e}")

                while client.is_connected:
                    await _process_pending_commands(name, client)
                    await asyncio.sleep(0.5)

        except Exception as e:
            print(f"[!] {name}: {e}")

        active_clients.pop(name, None)
        with lock:
            devices[name]["connected"] = False
        print(f"[-] {name} disconnected — retry in {RECONNECT_SEC}s")
        await asyncio.sleep(RECONNECT_SEC)

        # Refresh device object in case the Arduino rebooted/changed address
        current = await _refresh_device(name, current)


# ─────────────────────── COMMAND HELPERS ───────────────────────

def _parse_settings(name, settings_str):
    """Parse settings response like 'AD:16700 RI:500' into device state."""
    global current_antenna_delay
    with lock:
        if name not in devices:
            return
        for part in settings_str.strip().split():
            if part.startswith("AD:"):
                try:
                    ad_val = int(part[3:])
                    devices[name]["settings"]["antenna_delay"] = ad_val
                    if ad_val and ad_val != current_antenna_delay:
                        old_ad = current_antenna_delay
                        current_antenna_delay = ad_val
                        # Close current log CSVs and open new ones in the new AD folder
                        tag_csv.close()
                        anchor_csv.close()
                        _open_log_csvs()
                        print(f"  Antenna delay changed {old_ad} → {ad_val}, logs rotated")
                except ValueError:
                    pass
            elif part.startswith("RI:"):
                try:
                    devices[name]["settings"]["range_interval"] = int(part[3:])
                except ValueError:
                    pass


async def _process_pending_commands(name, client):
    """Send any queued commands for this device over BLE."""
    to_send = []
    with cmd_lock:
        remaining = []
        for dev, cmd in pending_commands:
            if dev == name:
                to_send.append(cmd)
            else:
                remaining.append((dev, cmd))
        pending_commands.clear()
        pending_commands.extend(remaining)

    for cmd in to_send:
        try:
            print(f"[CMD] Sending to {name}: {cmd}")
            await client.write_gatt_char(CMD_CHAR_UUID, cmd.encode("utf-8"))
            await asyncio.sleep(0.3)
            # Read back response
            resp = await client.read_gatt_char(CMD_CHAR_UUID)
            resp_str = resp.decode("utf-8", errors="replace")
            print(f"[CMD] {name} response: {resp_str}")
            _parse_settings(name, resp_str)
        except Exception as e:
            print(f"[CMD] Error sending to {name}: {e}")


# ─────────────────────── FLASK DASHBOARD ───────────────────────
app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    resp = make_response(render_template("dashboard.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/state")
def api_state():
    """JSON snapshot for the dashboard to poll."""
    with lock:
        result = {}
        for name, d in devices.items():
            hist = list(d["history"])[-50:]  # last 50 for the API
            result[name] = {
                "type": d["type"],
                "connected": d["connected"],
                "addr": d["addr"],
                "connect_time": d["connect_time"],
                "packet_count": d["packet_count"],
                "latest": d["latest"],
                "history": hist,
                "settings": d.get("settings", {}),
            }
    return jsonify({
        "session_start": session_start,
        "devices": result,
        "tag_log": str(tag_csv_path),
        "anchor_log": str(anchor_csv_path),
        "antenna_delay": current_antenna_delay,
    })


@app.route("/api/command", methods=["POST"])
def api_command():
    """Send a command to a device over BLE.

    JSON body: { "device": "T1", "command": "AD:16700" }
    Supported commands:
      AD:<value>  - Set antenna delay register (0 to 65535)
      RI:<value>  - Set range interval in ms (50 to 5000, tags only)
      ST          - Query current settings
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    device = data.get("device", "").strip()
    command = data.get("command", "").strip()

    if not device or not command:
        return jsonify({"error": "device and command fields required"}), 400

    with lock:
        if device not in devices:
            return jsonify({"error": f"Unknown device: {device}"}), 404
        if not devices[device]["connected"]:
            return jsonify({"error": f"{device} is not connected"}), 409

    # Validate command format
    if command.startswith("AD:"):
        try:
            val = int(command[3:])
            if val < 0 or val > 65535:
                return jsonify({"error": "Antenna delay must be 0 to 65535"}), 400
        except ValueError:
            return jsonify({"error": "AD value must be an integer"}), 400
    elif command.startswith("RI:"):
        try:
            val = int(command[3:])
            if val < 50 or val > 5000:
                return jsonify({"error": "Range interval must be 50 to 5000 ms"}), 400
        except ValueError:
            return jsonify({"error": "RI value must be an integer"}), 400
    elif command != "ST":
        return jsonify({"error": f"Unknown command: {command}"}), 400

    # Queue the command for the BLE thread to process
    with cmd_lock:
        pending_commands.append((device, command))

    return jsonify({"status": "queued", "device": device, "command": command})


@app.route("/api/set_all_delay", methods=["POST"])
def api_set_all_delay():
    """Convenience: set antenna delay on ALL connected devices at once.

    JSON body: { "delay": 16700 }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    delay_val = data.get("delay")
    if delay_val is None:
        return jsonify({"error": "delay field required"}), 400

    try:
        delay_val = int(delay_val)
        if delay_val < 0 or delay_val > 65535:
            return jsonify({"error": "Delay must be 0 to 65535"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Delay must be an integer"}), 400

    queued = []
    with lock:
        connected = [n for n, d in devices.items() if d["connected"]]

    cmd = f"AD:{delay_val}"
    with cmd_lock:
        for name in connected:
            pending_commands.append((name, cmd))
            queued.append(name)

    return jsonify({"status": "queued", "command": cmd, "devices": queued})


# ─────────────────────── DATA COLLECTION API ───────────────────────

@app.route("/api/collection/start", methods=["POST"])
def api_collection_start():
    """Start labelled data collection.

    JSON body: { "true_dist_m": 1.0, "angle_deg": 0, "notes": "", "target_samples": 500 }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    try:
        td = float(data.get("true_dist_m", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "true_dist_m must be a number"}), 400

    if td < 0:
        return jsonify({"error": "true_dist_m must be >= 0"}), 400

    angle = float(data.get("angle_deg", 0))
    notes = str(data.get("notes", ""))
    target = int(data.get("target_samples", 500))
    if target < 1:
        target = 500

    with collection_lock:
        collection["active"] = True
        collection["true_dist_m"] = td
        collection["angle_deg"] = angle
        collection["notes"] = notes
        collection["session_id"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        collection["target_samples"] = target
        collection["samples"] = []

    # Rotate log CSVs so this run gets its own files labelled with distance/angle
    tag_csv.close()
    anchor_csv.close()
    label = f"{td}m_{int(angle)}deg"
    _open_log_csvs(label=label)

    return jsonify({
        "status": "started",
        "true_dist_m": td,
        "angle_deg": angle,
        "target_samples": target,
        "session_id": collection["session_id"],
    })


@app.route("/api/collection/stop", methods=["POST"])
def api_collection_stop():
    """Stop collection and save to dataset CSV."""
    with collection_lock:
        collection["active"] = False

    saved = _save_collection()

    # Rotate back to generic (unlabelled) log files
    tag_csv.close()
    anchor_csv.close()
    _open_log_csvs()

    with collection_lock:
        samples = collection["samples"]
        if samples:
            dists = [s["distance_m"] for s in samples
                     if isinstance(s["distance_m"], (int, float)) and not math.isnan(s["distance_m"])]
            errors = [s["error_m"] for s in samples
                      if isinstance(s["error_m"], (int, float)) and not math.isnan(s["error_m"])]
            stats = {
                "count": len(samples),
                "mean_dist": float(np.mean(dists)) if dists else None,
                "std_dist": float(np.std(dists)) if dists else None,
                "mean_error": float(np.mean(errors)) if errors else None,
                "true_dist_m": collection["true_dist_m"],
                "angle_deg": collection["angle_deg"],
            }
        else:
            stats = {"count": 0}

    return jsonify({"status": "stopped", "saved": saved, "stats": stats})


@app.route("/api/collection/status")
def api_collection_status():
    """Get current collection state and live stats."""
    with collection_lock:
        active = collection["active"]
        samples = list(collection["samples"])
        td = collection["true_dist_m"]
        angle = collection["angle_deg"]
        target = collection["target_samples"]
        sid = collection["session_id"]

    count = len(samples)
    dists = [s["distance_m"] for s in samples
             if isinstance(s["distance_m"], (int, float)) and not math.isnan(s["distance_m"])]
    errors = [s["error_m"] for s in samples
              if isinstance(s["error_m"], (int, float)) and not math.isnan(s["error_m"])]

    return jsonify({
        "active": active,
        "session_id": sid,
        "true_dist_m": td,
        "angle_deg": angle,
        "target_samples": target,
        "count": count,
        "mean_dist": float(np.mean(dists)) if dists else None,
        "std_dist": float(np.std(dists)) if dists else None,
        "mean_error": float(np.mean(errors)) if errors else None,
        "last_5_dists": [s["distance_m"] for s in samples[-5:]],
    })


@app.route("/api/collection/dataset_info")
def api_collection_dataset_info():
    """Show info about the saved calibration dataset."""
    cal_path = _cal_dataset_path()
    if not cal_path.exists():
        return jsonify({"exists": False, "total_samples": 0, "distances": []})
    df = pd.read_csv(cal_path)
    dist_summary = []
    for td in sorted(df["true_dist_m"].unique()):
        sub = df[df["true_dist_m"] == td]
        dist_summary.append({
            "true_dist_m": float(td),
            "count": len(sub),
            "angles": sorted(sub["angle_deg"].unique().tolist()),
        })
    return jsonify({
        "exists": True,
        "total_samples": len(df),
        "distances": dist_summary,
    })


def run_flask():
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)


# ─────────────────────── MAIN BLE LOOP ───────────────────────

async def ble_main():
    """
    Runs forever.  Every RESCAN_INTERVAL seconds it scans for UWB devices
    and spawns a persistent handler for any device not yet managed (or whose
    previous handler crashed).  This means:
      - Devices powered on after startup are picked up automatically.
      - The dashboard never exits due to "no devices found".
      - Each handler reconnects indefinitely on its own; ble_main only
        handles late-joiners and crashed handler recovery.
    """
    print(f"Scanning every {RESCAN_INTERVAL}s for: {', '.join(ALL_NAMES)}")
    print("Dashboard stays running until Ctrl+C — devices may join at any time.\n")

    managed: dict = {}   # name → asyncio.Task

    # Graceful shutdown on Ctrl+C
    loop      = asyncio.get_event_loop()
    stop_flag = asyncio.Event()

    def _stop():
        print("\nShutting down…")
        for t in managed.values():
            t.cancel()
        stop_flag.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except (NotImplementedError, RuntimeError):
            pass   # Windows / Jupyter may not support signal handlers

    while not stop_flag.is_set():
        print(f"[SCAN] Scanning {SCAN_TIMEOUT}s…")
        try:
            found = await BleakScanner.discover(timeout=SCAN_TIMEOUT)
        except Exception as e:
            print(f"[SCAN] Error: {e}  — retrying in {RESCAN_INTERVAL}s")
            try:
                await asyncio.wait_for(stop_flag.wait(), timeout=RESCAN_INTERVAL)
            except asyncio.TimeoutError:
                pass
            continue

        targets = [d for d in found if d.name in ALL_NAMES]
        targets.sort(key=lambda d: d.name)

        if targets:
            print(f"[SCAN] Visible: {', '.join(d.name for d in targets)}")
        else:
            print("[SCAN] No UWB devices visible — will retry")

        for d in targets:
            task = managed.get(d.name)
            if task is None or task.done():
                handler = handle_tag if d.name in TAG_NAMES else handle_anchor
                managed[d.name] = asyncio.create_task(handler(d))
                print(f"[SCAN] Started handler for {d.name}")

        # Wait for the next scan interval (or until stop requested)
        try:
            await asyncio.wait_for(stop_flag.wait(), timeout=RESCAN_INTERVAL)
        except asyncio.TimeoutError:
            pass


def main():
    print("=" * 60)
    print("  UWB BLE Dashboard")
    print(f"  Session: {session_start}")
    print(f"  Logs:    {LOG_DIR.resolve()}")
    print(f"  Web UI:  http://0.0.0.0:{FLASK_PORT}")
    print("=" * 60 + "\n")

    # Start Flask in a daemon thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Run BLE event loop on main thread.
    # In notebooks (already-running loop), schedule a task instead of asyncio.run().
    try:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(ble_main())
        else:
            print("Detected running asyncio loop; scheduling ble_main() task.")
            return running_loop.create_task(ble_main())
    except KeyboardInterrupt:
        pass
    finally:
        tag_csv.close()
        anchor_csv.close()
        print(f"\nLogs saved:")
        print(f"  Tags:    {tag_csv_path}")
        print(f"  Anchors: {anchor_csv_path}")


if __name__ == "__main__":
    main()
