"""Download C-MAPSS and extract the FD001 files into data/raw/.

    python -m data.download
"""

import io
import os
import urllib.request
import zipfile

import config

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
FILES = tuple(f"{kind}_{config.DATASET}.txt" for kind in ("train", "test", "RUL"))


def raw_path(name):
    return os.path.join(config.RAW_DIR, name)


def download(force=False):
    """Fetch the NASA archive and extract the three FD001 files. Skips if they already exist."""
    if not force and all(os.path.exists(raw_path(f)) for f in FILES):
        return
    os.makedirs(config.RAW_DIR, exist_ok=True)
    with urllib.request.urlopen(URL) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    # The NASA archive wraps CMAPSSData.zip inside another zip
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    for name in FILES:
        with open(raw_path(name), "wb") as f:
            f.write(inner.read(name))


if __name__ == "__main__":
    download(force=True)
    print(f"wrote {', '.join(FILES)} to {config.RAW_DIR}")
