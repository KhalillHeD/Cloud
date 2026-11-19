#!/usr/bin/env python3
"""Download METR-LA into data/ (block 1).

Files:
  metr-la.h5                  speeds, 34272 x 207 (5-min slots, 2012-03-01 .. 2012-06-27, mph)
  graph_sensor_locations.csv  sensor_id, latitude, longitude
  distances_la_2012.csv       from, to, cost (road distance in metres)

The two CSVs come from the DCRNN repository. The DCRNN authors host metr-la.h5 on Google
Drive (not scriptable), so mirrors are tried in order. Each download is checked (size,
HDF5 signature, expected shape) before it replaces anything. If every mirror fails,
download metr-la.h5 by hand from the DCRNN README link and put it in data/.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DCRNN = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/"
FILES = {
    "graph_sensor_locations.csv": [DCRNN + "graph_sensor_locations.csv"],
    "distances_la_2012.csv": [DCRNN + "distances_la_2012.csv"],
    "metr-la.h5": [
        "https://huggingface.co/datasets/MintBruce/SkyTraffic/resolve/main/metr-la.h5",
    ],
}
MIN_SIZE = {"metr-la.h5": 50_000_000, "graph_sensor_locations.csv": 5_000, "distances_la_2012.csv": 100_000}
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "traffic-analytics-downloader"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    verify(tmp, dest.name)
    tmp.replace(dest)


def verify(path: Path, name: str) -> None:
    size = path.stat().st_size
    if size < MIN_SIZE[name]:
        raise ValueError(f"{name}: only {size} bytes")
    if name.endswith(".h5"):
        with open(path, "rb") as f:
            if f.read(8) != HDF5_MAGIC:
                raise ValueError(f"{name}: not an HDF5 file")
        sys.path.insert(0, str(ROOT / "common"))
        sys.path.insert(0, str(ROOT))
        from ml.dataset import read_metr_la_h5
        df = read_metr_la_h5(path)
        if df.shape != (34272, 207):
            raise ValueError(f"{name}: unexpected shape {df.shape}")
    else:
        head = path.read_text(encoding="utf-8").splitlines()[0].lower()
        if "," not in head:
            raise ValueError(f"{name}: not a CSV")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download again even if present")
    args = ap.parse_args(argv)
    DATA.mkdir(exist_ok=True)
    failed = []
    for name, urls in FILES.items():
        dest = DATA / name
        if dest.exists() and not args.force:
            try:
                verify(dest, name)
                print(f"  {name}: present")
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"  {name}: present but invalid ({exc}), downloading again")
        for url in urls:
            try:
                print(f"  {name} <- {url}")
                fetch(url, dest)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"    failed: {exc}")
        else:
            failed.append(name)
    if failed:
        print("Could not download: " + ", ".join(failed))
        print("Get them from https://github.com/liyaguang/DCRNN (README > Data Preparation) into data/.")
        return 1
    print("METR-LA ready in data/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
