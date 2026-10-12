"""CWRU bearing dataset: file list (12 kHz, drive-end accelerometer) and downloader.

File numbers verified against the official pages:
  https://engineering.case.edu/bearingdatacenter/normal-baseline-data
  https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data

Classes: normal, inner (inner race), ball, outer (outer race, sensor at 6:00).
Fault sizes: 0.007, 0.014, 0.021 inch. Motor loads: 0-3 HP.
"""
import os
import urllib.request

BASE_URL = "https://engineering.case.edu/sites/default/files/"
CLASSES = ["normal", "inner", "ball", "outer"]
FS_NATIVE = 12000  # Hz

# load (HP) -> file number
_NORMAL = {0: 97, 1: 98, 2: 99, 3: 100}
_FAULTS = {
    "inner": {
        0.007: {0: 105, 1: 106, 2: 107, 3: 108},
        0.014: {0: 169, 1: 170, 2: 171, 3: 172},
        0.021: {0: 209, 1: 210, 2: 211, 3: 212},
    },
    "ball": {
        0.007: {0: 118, 1: 119, 2: 120, 3: 121},
        0.014: {0: 185, 1: 186, 2: 187, 3: 188},
        0.021: {0: 222, 1: 223, 2: 224, 3: 225},
    },
    "outer": {
        0.007: {0: 130, 1: 131, 2: 132, 3: 133},
        0.014: {0: 197, 1: 198, 2: 199, 3: 200},
        0.021: {0: 234, 1: 235, 2: 236, 3: 237},
    },
}


def file_table():
    """Return a list of dicts: file_id, label (class index), cls, size (inch, 0 for normal), load (HP)."""
    rows = []
    for load, fid in _NORMAL.items():
        rows.append(dict(file_id=fid, label=CLASSES.index("normal"), cls="normal", size=0.0, load=load))
    for cls, by_size in _FAULTS.items():
        for size, by_load in by_size.items():
            for load, fid in by_load.items():
                rows.append(dict(file_id=fid, label=CLASSES.index(cls), cls=cls, size=size, load=load))
    return rows


def download(file_id, raw_dir):
    """Download one .mat file if not already present. Returns the local path."""
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{file_id}.mat")
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        return path
    url = f"{BASE_URL}{file_id}.mat"
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
        f.write(r.read())
    return path
