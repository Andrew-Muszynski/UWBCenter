#!/usr/bin/env python3
"""
collect_session.py — Guided Data Collection Session
=====================================================
Walks through a structured calibration campaign, one distance at a time.
Talks to the uwb_dashboard.py REST API to start/stop labelled collection.

Usage:
    1. Start the dashboard:  ./start.sh
    2. Run this script:      python3 collect_session.py

The script will:
  - Verify the dashboard is reachable and devices are connected
  - Prompt you through each distance in the plan
  - Auto-collect 500 samples (~25 s) at each distance
  - Show live progress and basic stats after each distance
  - Print a final summary when done

Press Ctrl+C at any time to abort (already-saved distances are safe).
"""

import sys
import time
import json
import signal

try:
    import requests
except ImportError:
    print("ERROR: requests not installed.  pip install requests")
    sys.exit(1)

# ─────────────────────── CONFIG ───────────────────────────────────────────────

DASHBOARD_BASE = "http://localhost:5050"
API_STATE      = f"{DASHBOARD_BASE}/api/state"
API_COL_START  = f"{DASHBOARD_BASE}/api/collection/start"
API_COL_STOP   = f"{DASHBOARD_BASE}/api/collection/stop"
API_COL_STATUS = f"{DASHBOARD_BASE}/api/collection/status"
API_DATASET    = f"{DASHBOARD_BASE}/api/collection/dataset_info"

# Distances to collect.  Option C: whole-foot plan (classroom-feasible).
# Feet are converted to metres for the dataset (true_dist_m stays metric).
DISTANCES_FT = [1, 2, 3, 5, 8, 12, 20]
DISTANCES = [round(ft * 0.3048, 3) for ft in DISTANCES_FT]
# → [0.305, 0.610, 0.914, 1.524, 2.438, 3.658, 6.096] m

ANGLE_DEG       = 0        # fixed orientation for baseline campaign
SAMPLES_PER_DIST = 500     # 500 samples ≈ 25 s at 20 Hz
POLL_INTERVAL    = 2.0     # seconds between status polls during collection


# ─────────────────────── HELPERS ──────────────────────────────────────────────

def api_get(url, label=""):
    """GET with error handling."""
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        print(f"\n  ERROR: Cannot reach dashboard at {DASHBOARD_BASE}")
        print("  Make sure ./start.sh has been run and devices are connected.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR ({label}): {e}")
        sys.exit(1)


def api_post(url, body, label=""):
    """POST JSON with error handling."""
    try:
        r = requests.post(url, json=body, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        print(f"\n  ERROR: Lost connection to dashboard during {label}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR ({label}): {e}")
        sys.exit(1)


def check_dashboard():
    """Verify dashboard is up and at least one tag is connected."""
    state = api_get(API_STATE, "dashboard check")
    devs = state.get("devices", {})
    tags = {n: d for n, d in devs.items() if d.get("type") == "tag" and d.get("connected")}
    anchors = {n: d for n, d in devs.items() if d.get("type") == "anchor" and d.get("connected")}

    if not tags:
        print("  WARNING: No tags connected!  Ranging won't produce data.")
        print("  Check that your tag is powered on and in BLE range of the Pi.\n")
        ans = input("  Continue anyway? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit(0)
    if not anchors:
        print("  WARNING: No anchors connected!  SS-TWR needs both tag and anchor.\n")
        ans = input("  Continue anyway? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit(0)

    ad_vals = set()
    for d in devs.values():
        ad = d.get("settings", {}).get("antenna_delay", 0)
        if ad:
            ad_vals.add(ad)

    print(f"  Tags connected    : {', '.join(tags.keys()) or 'NONE'}")
    print(f"  Anchors connected : {', '.join(anchors.keys()) or 'NONE'}")
    if ad_vals:
        print(f"  Antenna delay(s)  : {', '.join(str(v) for v in sorted(ad_vals))}")
    print()
    return tags, anchors


def collect_one_distance(dist_m, angle_deg, target_n):
    """Run one collection session and return stats dict."""
    body = {
        "true_dist_m": dist_m,
        "angle_deg": angle_deg,
        "target_samples": target_n,
        "notes": f"baseline_session",
    }
    resp = api_post(API_COL_START, body, f"start {dist_m}m")
    sid = resp.get("session_id", "?")
    print(f"  Session {sid} started — collecting {target_n} samples …")

    # Poll until done
    while True:
        time.sleep(POLL_INTERVAL)
        status = api_get(API_COL_STATUS, "poll")
        count = status.get("count", 0)
        mean_d = status.get("mean_dist")
        mean_e = status.get("mean_error")
        active = status.get("active", False)

        bar_len = int(count / target_n * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pct = count / target_n * 100

        mean_str = f"mean={mean_d:.3f}m" if mean_d is not None else "mean=—"
        err_str  = f"err={mean_e:+.3f}m" if mean_e is not None else "err=—"
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  n={count}  {mean_str}  {err_str}   ")
        sys.stdout.flush()

        if not active or count >= target_n:
            break

    # Stop and save
    stop_resp = api_post(API_COL_STOP, {}, f"stop {dist_m}m")
    stats = stop_resp.get("stats", {})
    print()  # newline after progress bar
    return stats


def print_summary(results):
    """Print a table of all collected distances."""
    print("\n" + "=" * 62)
    print("  COLLECTION SUMMARY")
    print("=" * 62)
    print(f"  {'Dist (m)':>10}  {'Samples':>8}  {'Mean (m)':>10}  {'Error (m)':>10}  {'Std (m)':>10}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    total = 0
    for r in results:
        td = r.get("true_dist_m", "?")
        n  = r.get("count", 0)
        md = r.get("mean_dist")
        me = r.get("mean_error")
        sd = r.get("std_dist")
        total += n
        md_s = f"{md:.3f}" if md is not None else "—"
        me_s = f"{me:+.3f}" if me is not None else "—"
        sd_s = f"{sd:.3f}" if sd is not None else "—"
        print(f"  {td:>10}  {n:>8}  {md_s:>10}  {me_s:>10}  {sd_s:>10}")
    print(f"  {'':>10}  {'─'*8}")
    print(f"  {'TOTAL':>10}  {total:>8}")
    print()

    # Show dataset info
    info = api_get(API_DATASET, "dataset info")
    print(f"  Cal dataset total : {info.get('total_samples', '?')} samples")
    if info.get("distances"):
        for d in info["distances"]:
            print(f"    {d['true_dist_m']:.1f} m : {d['count']} samples  angles={d['angles']}")
    print()


# ─────────────────────── MAIN ─────────────────────────────────────────────────

def main():
    # Handle Ctrl+C gracefully
    def sigint_handler(sig, frame):
        print("\n\n  Interrupted — stopping any active collection …")
        try:
            requests.post(API_COL_STOP, json={}, timeout=3)
        except Exception:
            pass
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    print()
    print("=" * 62)
    print("  UWB BASELINE DATA COLLECTION SESSION")
    print("=" * 62)
    print(f"  Distances : {DISTANCES_FT} ft  →  {DISTANCES} m")
    print(f"  Angle     : {ANGLE_DEG}°  (fixed)")
    print(f"  Samples   : {SAMPLES_PER_DIST} per distance")
    print(f"  Est. time : ~{len(DISTANCES) * 1.5:.0f} min  (25s collect + setup each)")
    print()

    # Pre-flight checks
    print("─── Pre-flight checks ───")
    check_dashboard()

    # Check existing dataset
    info = api_get(API_DATASET, "dataset info")
    existing = info.get("total_samples", 0)
    if existing > 0:
        print(f"  Existing cal dataset has {existing} samples.")
        existing_dists = [d["true_dist_m"] for d in info.get("distances", [])]
        if existing_dists:
            print(f"  Distances already covered: {existing_dists}")
        print("  New samples will be APPENDED (not overwritten).\n")

    # Confirm
    input("  Press ENTER to begin (Ctrl+C to abort) … ")
    print()

    results = []
    for i, (dist, ft) in enumerate(zip(DISTANCES, DISTANCES_FT)):
        print(f"─── Distance {i+1}/{len(DISTANCES)}: {ft} ft  ({dist} m) ───")
        print(f"  ACTION: Place tag at exactly {ft} ft ({dist} m) from anchor.")
        input(f"  Press ENTER when tag is positioned at {ft} ft … ")

        stats = collect_one_distance(dist, ANGLE_DEG, SAMPLES_PER_DIST)
        results.append(stats)

        # Quick sanity check
        me = stats.get("mean_error")
        if me is not None and abs(me) > 2.0:
            print(f"  ⚠  Large mean error ({me:+.2f} m) — double-check your setup.")
        elif me is not None:
            print(f"  ✓  Looks good (error = {me:+.3f} m)")
        print()

    print_summary(results)
    print("  Done! Open uwb_ble_calibration.py to train your model.")
    print("  The data is in logs/AD_<delay>/ble_cal_dataset.csv\n")


if __name__ == "__main__":
    main()
