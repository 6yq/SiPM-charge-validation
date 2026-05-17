#!/usr/bin/env python3
"""Download and extract Hamamatsu PCB6 Light histogram data from Zenodo."""
import argparse
import hashlib
import io
import os
import urllib.request
import zipfile

ZENODO_URL="https://zenodo.org/api/records/10014537/files/peakotron-1.0.0.zip/content"
EXPECTED_MD5="3419a012f2a5b59585247d1a7ffa1809"
DATA_PREFIX="peakotron-1.0.0/data/hamamatsu_pcb6/Light/"


def download(url, dest_path):
    print(f"Downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"Saved to {dest_path}", flush=True)


def verify_md5(path, expected):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        raise RuntimeError(f"MD5 mismatch: expected {expected}, got {got}")
    print(f"MD5 OK: {got}", flush=True)


def extract(zip_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith(DATA_PREFIX) and m.endswith(".txt")]
        for member in members:
            fname = os.path.basename(member)
            dest = os.path.join(out_dir, fname)
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"  extracted: {fname}", flush=True)
    print(f"Extracted {len(members)} files to {out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out-dir", default="data/", help="Output directory")
    parser.add_argument("--zip-cache", default="/tmp/peakotron.zip", help="Local zip cache path")
    args = parser.parse_args()

    if not os.path.exists(args.zip_cache):
        download(ZENODO_URL, args.zip_cache)
    else:
        print(f"Using cached zip: {args.zip_cache}", flush=True)

    verify_md5(args.zip_cache, EXPECTED_MD5)
    extract(args.zip_cache, args.out_dir)


if __name__ == "__main__":
    main()
