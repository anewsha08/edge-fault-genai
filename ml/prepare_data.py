"""Build the feature dataset from the CWRU bearing data (or synthetic data for a quick test).

Usage:
  python ml/prepare_data.py                    # download CWRU, build ml/data/features.npz
  python ml/prepare_data.py --synthetic        # no download; fake vibration data to test the pipeline
  python ml/prepare_data.py --fs 3200          # resample to 3.2 kHz (e.g. ADXL345 max output rate)

Each row of the output carries the recording it came from (file_id), motor load, and fault size, so
train.py can split by recording/load instead of by random window (random window splits leak:
neighbouring windows overlap, so test accuracy looks far better than it really is).
"""
import argparse
import math
import os

import numpy as np

import cwru_files as cw
from features import window_signal, windows_to_features, feature_names


def load_de_signal(mat_path):
    from scipy.io import loadmat
    m = loadmat(mat_path)
    keys = [k for k in m if k.endswith("DE_time")]
    if not keys:
        raise KeyError(f"No *_DE_time variable in {mat_path}; keys: {[k for k in m if not k.startswith('__')]}")
    return np.asarray(m[keys[0]], dtype=np.float64).ravel()


def resample(x, fs_from, fs_to):
    if fs_to == fs_from:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(int(fs_from), int(fs_to))
    return resample_poly(x, int(fs_to) // g, int(fs_from) // g)


def synthetic_signal(cls, size, load, fs, seconds, rng):
    """Crude stand-in for CWRU: noise + shaft tone, plus decaying resonance bursts for faults."""
    n = int(fs * seconds)
    t = np.arange(n) / fs
    x = 0.05 * rng.standard_normal(n) + 0.03 * np.sin(2 * np.pi * 30 * t)
    x *= 1 + 0.05 * load
    rate = {"inner": 162.0, "ball": 141.0, "outer": 107.0}.get(cls)
    if rate:
        amp = {"inner": 0.5, "ball": 0.25, "outer": 0.4}[cls] * (size / 0.007) ** 0.5
        f_res = {"inner": 3200.0, "ball": 2400.0, "outer": 3800.0}[cls]
        f_res = min(f_res, 0.45 * fs)
        period = int(fs / rate)
        k = np.arange(int(0.004 * fs))
        burst = np.exp(-k / (0.0007 * fs)) * np.sin(2 * np.pi * f_res * k / fs)
        for s in range(0, n - len(burst), period):
            jitter = int(rng.integers(-3, 4))
            s2 = max(0, s + jitter)
            x[s2:s2 + len(burst)] += amp * burst * (0.7 + 0.6 * rng.random())
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="ml/data/raw")
    ap.add_argument("--out", default="ml/data/features.npz")
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--fs", type=int, default=cw.FS_NATIVE, help="target sampling rate in Hz")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    X, y, file_id, load, size = [], [], [], [], []
    rng = np.random.default_rng(0)
    rows = cw.file_table()
    for i, r in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] file {r['file_id']} {r['cls']} size={r['size']} load={r['load']}")
        if args.synthetic:
            sig = synthetic_signal(r["cls"], r["size"], r["load"], cw.FS_NATIVE, 10, rng)
        else:
            sig = load_de_signal(cw.download(r["file_id"], args.raw_dir))
        sig = resample(sig, cw.FS_NATIVE, args.fs)
        wins = window_signal(sig, args.win, args.hop)
        if len(wins) == 0:
            print("  (too short, skipped)")
            continue
        feats = windows_to_features(wins)
        X.append(feats)
        n = len(feats)
        y += [r["label"]] * n
        file_id += [r["file_id"]] * n
        load += [r["load"]] * n
        size += [r["size"]] * n

    X = np.vstack(X)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out, X=X.astype(np.float32), y=np.array(y), file_id=np.array(file_id),
        load=np.array(load), size=np.array(size), classes=np.array(cw.CLASSES),
        feature_names=np.array(feature_names()), fs=args.fs, win=args.win, hop=args.hop,
        synthetic=args.synthetic,
    )
    counts = {c: int((np.array(y) == i).sum()) for i, c in enumerate(cw.CLASSES)}
    print(f"saved {args.out}: X={X.shape}, windows per class={counts}")


if __name__ == "__main__":
    main()
