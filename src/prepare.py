"""WESAD -> data/hrv_features.csv

Labels and signal both come from SX.pkl, which the dataset ships already
time-aligned. Two earlier sources were rejected:

  * SX_quest.csv + E4 IBI.csv — the wristband clock starts 18-26 min before the
    protocol (varies per subject), so those timings mislabel every window. This
    is what made stress look calmer than baseline in the original pipeline.
  * wrist PPG — usable, but the stress task involves standing and speaking, and
    the motion leaves HRV unusable there (the E4's own beat detector keeps only
    0-13% of beats during it).

Chest ECG is clean across all conditions. RR intervals are the same physiological
quantity the app derives from the camera, and a fingertip held still on a lens is
closer to this signal quality than a wrist in motion.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from features import compute

ECG_FS = 700
WINDOW_S, STEP_S = 60.0, 30.0
CONDITIONS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def beat_times(ecg):
    """Seconds of each R peak, via energy-envelope detection."""
    band = filtfilt(*butter(3, [5, 20], btype="band", fs=ECG_FS), ecg)
    window = int(0.1 * ECG_FS)
    energy = np.convolve(band**2, np.ones(window) / window, mode="same")
    peaks, _ = find_peaks(
        energy, distance=int(0.3 * ECG_FS), height=0.25 * np.percentile(energy, 95)
    )
    return peaks / ECG_FS


def segments(labels):
    """Contiguous (condition, start_s, end_s) runs of the four scored conditions."""
    edges = np.flatnonzero(np.diff(labels)) + 1
    for start, end in zip(np.r_[0, edges], np.r_[edges, len(labels)]):
        if labels[start] in CONDITIONS:
            yield CONDITIONS[labels[start]], start / ECG_FS, end / ECG_FS


def subject_rows(subject_dir):
    with open(subject_dir / f"{subject_dir.name}.pkl", "rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    ecg = np.asarray(data["signal"]["chest"]["ECG"], dtype=float).ravel()
    labels = np.asarray(data["label"]).ravel()
    del data

    beats = beat_times(ecg)
    for condition, start, end in segments(labels):
        for window_start in np.arange(start, end - WINDOW_S, STEP_S):
            in_window = (beats >= window_start) & (beats < window_start + WINDOW_S)
            row = compute(np.diff(beats[in_window]) * 1000.0)
            if row:
                yield {
                    "subject": subject_dir.name,
                    "condition": condition,
                    "stress": int(condition == "stress"),
                    "window_start_s": round(float(window_start), 1),
                    **row,
                }


def main():
    parser = argparse.ArgumentParser(description="Extract HRV features from WESAD")
    parser.add_argument("--data-dir", type=Path, required=True, help="WESAD root")
    parser.add_argument("--output", type=Path, default=Path("data/hrv_features.csv"))
    args = parser.parse_args()

    subjects = sorted(d for d in args.data_dir.iterdir() if (d / f"{d.name}.pkl").exists())
    rows = []
    for subject_dir in subjects:
        found = list(subject_rows(subject_dir))
        rows += found
        print(f"{subject_dir.name}: {len(found)} windows", flush=True)

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"\n{len(frame)} windows, {frame.subject.nunique()} subjects -> {args.output}")
    print(frame.groupby("condition")[["mean_hr", "rmssd", "lf_hf_ratio"]].mean().round(1))


if __name__ == "__main__":
    main()
