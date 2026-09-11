"""Download published benchmark inputs without overwriting existing data.

Run: python -m interactiondynamics.data.download_benchmarks
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


SOURCES = {
    "college_msg": ("CollegeMsg.txt.gz", "https://snap.stanford.edu/data/CollegeMsg.txt.gz"),
    "email_eu_core": ("email-Eu-core-temporal.txt.gz", "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz"),
    "sociopatterns": ("HighSchool2013_proximity_net.csv.gz", "https://sociopatterns.org/assets/data/HighSchool2013_proximity_net.csv.gz"),
    "metr_la": ("metr-la.h5", "https://drive.google.com/uc?id=1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC"),
    "pems_bay": ("pems-bay.h5", "https://drive.google.com/uc?id=1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq"),
}


def download(name: str, root: str | Path = "data") -> Path:
    filename, url = SOURCES[name]
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if not target.exists():
        part = directory / (filename + ".part")
        if "drive.google.com" in url:
            import gdown
            if gdown.download(url, str(part), quiet=False) is None:
                raise RuntimeError(f"Download failed: {url}")
        else:
            with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as output:
                shutil.copyfileobj(response, output)
        # Reject an HTML error/login page before publishing the file.
        with part.open("rb") as handle:
            magic = handle.read(8)
        if filename.endswith(".gz"):
            with gzip.open(part, "rb") as handle:
                while handle.read(1024 * 1024):
                    pass
        elif magic != b"\x89HDF\r\n\x1a\n":
            raise ValueError(f"Expected HDF5 data from {url}; retained {part} for inspection")
        part.replace(target)
    with target.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = directory / "download.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({"name": name, "url": url, "file": filename,
                                        "sha256": digest}, indent=2) + "\n")
    print(f"{name}: {target} ({target.stat().st_size:,} bytes)", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=list(SOURCES))
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    failed = []
    for name in args.names:
        if name not in SOURCES:
            parser.error(f"Unknown dataset {name}; choose from {list(SOURCES)}")
        try:
            download(name, args.root)
        except Exception as exc:
            print(f"FAILED {name}: {exc}", flush=True)
            failed.append(name)
    if failed:
        raise SystemExit(f"Failed downloads: {', '.join(failed)}")


if __name__ == "__main__":
    main()
