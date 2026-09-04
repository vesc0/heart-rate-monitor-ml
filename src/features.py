"""HRV features from RR intervals. Everything here is derivable from PPG alone,
so the phone camera can feed the same model."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt, welch

# Deliberately excluded as redundant: sdsd (~rmssd), hf_norm (=100-lf_norm),
# mean_rr (=60000/mean_hr).
FEATURES = (
    "sdnn", "median_rr", "cv_rr", "rmssd", "pnn50", "pnn20",
    "mean_hr", "std_hr", "min_hr", "max_hr", "hr_range",
    "lf_power", "hf_power", "lf_hf_ratio", "total_power", "lf_norm",
    "sd1", "sd2", "sd_ratio",
)

_BANDS = {"lf": (0.04, 0.15), "hf": (0.15, 0.40)}


def clean(rr):
    """Drop non-physiological and locally deviant intervals.

    Returns the surviving intervals plus a mask marking which neighbouring pairs
    were still adjacent in the original series, so successive-difference metrics
    are not computed across a gap.
    """
    ok = (rr >= 300) & (rr <= 2000)
    if ok.sum() >= 5:
        reference = np.full_like(rr, np.median(rr[ok]))
        reference[ok] = medfilt(rr[ok], 5)
        ok &= np.abs(rr - reference) <= 0.2 * reference
    kept = np.flatnonzero(ok)
    return rr[kept], np.diff(kept) == 1


def _spectral(rr):
    """LF/HF power from the RR series resampled onto a uniform 4 Hz grid."""
    zero = dict.fromkeys(("lf_power", "hf_power", "lf_hf_ratio", "total_power", "lf_norm"), 0.0)
    times = np.cumsum(rr) / 1000.0
    grid = np.arange(times[0], times[-1], 0.25)
    if len(grid) < 32:
        return zero

    series = interp1d(times, rr, kind="cubic", fill_value="extrapolate")(grid)
    freqs, psd = welch(series - series.mean(), fs=4.0, nperseg=min(len(series), 256))

    def power(lo, hi):
        band = (freqs >= lo) & (freqs < hi)
        return float(np.trapezoid(psd[band], freqs[band])) if band.any() else 0.0

    lf, hf = power(*_BANDS["lf"]), power(*_BANDS["hf"])
    total = lf + hf
    return {
        "lf_power": lf,
        "hf_power": hf,
        "lf_hf_ratio": lf / hf if hf else 0.0,
        "total_power": total,
        "lf_norm": lf / total * 100 if total else 0.0,
    }


def compute(rr, min_beats=30):
    """Feature dict for one window of RR intervals in ms, or None if too sparse."""
    rr, adjacent = clean(np.asarray(rr, dtype=float))
    if len(rr) < min_beats or not adjacent.any():
        return None

    diffs = np.diff(rr)[adjacent]
    hr = 60000.0 / rr
    sdnn = float(rr.std(ddof=1))
    sd1 = float(diffs.std(ddof=1) / np.sqrt(2))
    sd2 = float(np.sqrt(max(2 * sdnn**2 - sd1**2, 0.0)))

    return {
        "sdnn": sdnn,
        "median_rr": float(np.median(rr)),
        "cv_rr": sdnn / float(rr.mean()),
        "rmssd": float(np.sqrt(np.mean(diffs**2))),
        "pnn50": float(np.mean(np.abs(diffs) > 50) * 100),
        "pnn20": float(np.mean(np.abs(diffs) > 20) * 100),
        "mean_hr": float(hr.mean()),
        "std_hr": float(hr.std(ddof=1)),
        "min_hr": float(hr.min()),
        "max_hr": float(hr.max()),
        "hr_range": float(hr.max() - hr.min()),
        "sd1": sd1,
        "sd2": sd2,
        "sd_ratio": sd2 / sd1 if sd1 else 0.0,
        **_spectral(rr),
    }
