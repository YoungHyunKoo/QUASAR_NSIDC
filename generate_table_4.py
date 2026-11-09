"""
Search and download AMSR2 L1R files from JAXA G-Portal (new system, October 2026).

  * Search : G-Portal catalogue (CSW) at https://csw.gportal.jaxa.jp/csw
  * Download: G-Portal SFTP server with your private key, using the system
              `sftp` command (the same command that works in your terminal)

Usage:
    python amsr2_gportal_download.py 2024-01-01 2024-01-31

No `gportal` package is needed. Requirements: requests, and the `sftp` command.
"""

import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta

import requests

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SCRATCH = "/scratch/alpine/yoko2261/AMSR"
DATASET_IDS = ['11001002']        # L1R brightness temperature (see gportal.datasets())
BBOX = [-180, -90, 180, -50]      # lon_min, lat_min, lon_max, lat_max

CSW_URL = "https://csw.gportal.jaxa.jp/csw"

SFTP_HOST = "sftp.gportal.jaxa.jp"
SFTP_PORT = 22                                                  # same port as in your working sftp command
SFTP_USER = "kooala317"                             # same user as in your working sftp command
SFTP_KEY = os.path.expanduser("/home/yoko2261/.ssh/privatekey.key")     # same key as in your working sftp command

# Folder of the files on the SFTP server; {year} and {month} are filled in per file
REMOTE_DIR = "product/Standard/GCOM-W/GCOM-W.AMSR2/L1R/2/{year}/{month}"

SAVE_AS_NC = True                 # True: save as .nc (matches the gridding script); False: keep .h5
PAGE_SIZE = 100                   # search results per request
MAX_RETRIES = 3
RETRY_WAIT_S = 30
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


# ---------------------------------------------------------------------------
# 1. Search
# ---------------------------------------------------------------------------
def granule_name(feature):
    """Granule ID, e.g. GW1AM2_202401010105_192D_L1SGRTBR_2220220."""
    props = feature.get("properties", {})
    name = props.get("identifier") or props.get("gpp", {}).get("identifier")
    if not name:
        file_name = props.get("product", {}).get("fileName", "")
        name = os.path.splitext(os.path.basename(file_name))[0]
    return name or None


def search_day(day):
    """Return the list of granule names for one day (all result pages)."""
    params = {
        "service": "CSW", "version": "3.0.0", "request": "GetRecords",
        "outputFormat": "application/json",
        "datasetId": ",".join(DATASET_IDS),
        "bbox": ",".join(str(v) for v in BBOX),
        "startTime": f"{day:%Y-%m-%d}T00:00:00Z",
        "endTime": f"{day:%Y-%m-%d}T23:59:59Z",
        "count": PAGE_SIZE,
    }
    names, start = [], 1
    while True:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(CSW_URL, params={**params, "startIndex": start}, timeout=120)
                r.raise_for_status()
                page = r.json()
                break
            except (requests.exceptions.RequestException, ValueError) as e:
                print(f"  search attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(RETRY_WAIT_S * attempt)

        features = page.get("features", [])
        names += [n for n in (granule_name(f) for f in features) if n]

        info = page.get("properties", {})
        returned = info.get("numberOfRecordsReturned", len(features))
        matched = info.get("numberOfRecordsMatched", 0)
        start += returned
        if returned == 0 or start > matched:
            break
    return sorted(set(names))


# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------
def remote_path(name):
    """Path on the SFTP server, built from the date in the file name."""
    stamp = name.split("_")[1]                    # e.g. 202401010105
    folder = REMOTE_DIR.format(year=stamp[:4], month=stamp[4:6])
    return f"{folder}/{name}.h5"


def local_path(name):
    ext = ".nc" if SAVE_AS_NC else ".h5"
    year = name.split("_")[1][:4]
    return os.path.join(SCRATCH, year, name + ext)


def is_hdf5(path):
    try:
        with open(path, "rb") as f:
            return f.read(8) == HDF5_SIGNATURE
    except OSError:
        return False


def already_done(path):
    return os.path.exists(path) and os.path.getsize(path) > 0 and is_hdf5(path)


# ---------------------------------------------------------------------------
# 3. Download (one sftp session per batch of files)
# ---------------------------------------------------------------------------
def sftp_get_batch(names):
    """Download several files in one sftp session. Returns names that failed."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as batch:
        for name in names:
            out = local_path(name)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            # leading '-' : continue with the next file if one fails
            batch.write(f'-get "{remote_path(name)}" "{out}.part"\n')
        batch_file = batch.name

    cmd = ["sftp", "-i", SFTP_KEY, "-P", str(SFTP_PORT), "-b", batch_file,
           f"{SFTP_USER}@{SFTP_HOST}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(batch_file)
    if result.stderr.strip():
        print("  sftp messages:\n   ", result.stderr.strip().replace("\n", "\n    "))

    failed = []
    for name in names:
        out = local_path(name)
        tmp = out + ".part"
        if is_hdf5(tmp):
            os.replace(tmp, out)
            print(f"  downloaded: {out}")
        else:
            if os.path.exists(tmp):
                os.remove(tmp)
            failed.append(name)
    return failed


def download(names):
    todo = []
    for name in names:
        if already_done(local_path(name)):
            print(f"  exists, skipping: {local_path(name)}")
        else:
            todo.append(name)

    for attempt in range(1, MAX_RETRIES + 1):
        if not todo:
            return
        todo = sftp_get_batch(todo)
        if todo and attempt < MAX_RETRIES:
            print(f"  {len(todo)} file(s) failed, retrying in {RETRY_WAIT_S * attempt} s")
            time.sleep(RETRY_WAIT_S * attempt)

    for name in todo:
        print(f"  GIVING UP: {name}  (remote path: {remote_path(name)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python amsr2_gportal_download.py YYYY-MM-DD YYYY-MM-DD")
    day = datetime.strptime(sys.argv[1], "%Y-%m-%d")
    last = datetime.strptime(sys.argv[2], "%Y-%m-%d")

    while day <= last:
        try:
            names = search_day(day)
        except Exception as e:
            print(f"{day:%Y-%m-%d}: search failed ({e}), skipping this day")
            names = []
        print(f"{day:%Y-%m-%d}: {len(names)} granules")
        if names:
            download(names)
        day += timedelta(days=1)


if __name__ == "__main__":
    main()
    
