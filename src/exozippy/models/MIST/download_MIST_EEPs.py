import shutil
import tarfile
from pathlib import Path

import requests

from .parse_MIST_EEP_filenames import _generate_MIST_EEP_url

# -------------------------------------------------------------------
# Download Files
# -------------------------------------------------------------------

# Seconds of silence before the socket errors. Not a budget for the whole
# transfer -- these tarballs are large, but a healthy one delivers a chunk
# continuously. Without it a stalled peer hangs the whole 186 GB workflow.
_HTTP_TIMEOUT = 60


def _flatten_single_top_level_entry(folder: Path) -> None:
    """If `folder` holds exactly one directory, lift its contents up.

    MIST ships each grid point as a tarball with its own top-level
    directory (``MIST_v2.5_feh_..._EEPS/``), while every consumer here
    addresses a grid point by the shorter folder name
    (``feh_p000_afe_p0_vvcrit0.0``, see generate_MIST_EEP_Tables). Lifting
    the wrapper away makes the two agree whichever way the archive is
    laid out, including the nested ``eeps/`` form the readers already
    handle.
    """
    entries = list(folder.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        return
    inner = entries[0]
    for item in list(inner.iterdir()):
        item.rename(folder / item.name)
    inner.rmdir()


def download_tarfiles(url, dest_folder="temp_files"):
    """Download one MIST tarball and extract it into `dest_folder`.

    Every step is staged, because the caller's skip-check is simply
    "does dest_folder exist": a dest_folder that exists but is half
    populated is the one state that must not be reachable. So the body
    lands on ``archive.tar.gz.part`` and is renamed only once the transfer
    finished; the extraction happens in a sibling staging directory; and
    only a complete extraction is renamed onto `dest_folder`. Anything
    that fails takes the staging directory with it.
    """
    dest = Path(dest_folder)
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_name(dest.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    tar_path = staging / "archive.tar.gz"
    part_path = staging / "archive.tar.gz.part"
    try:
        # 1. Download the file safely in chunks
        with requests.get(url, stream=True, timeout=_HTTP_TIMEOUT) as response:
            response.raise_for_status()
            with open(part_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        part_path.replace(tar_path)

        # 2. Extract (untar) the files
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=staging, filter="data")
        tar_path.unlink()

        _flatten_single_top_level_entry(staging)
        staging.replace(dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return dest


# -------------------------------------------------------------------
# Download All MISTv2.5 EEP Tracks
# -------------------------------------------------------------------

EEP_PATH = Path(
    "/Volumes/Data/EEP_Tracks/MISTv2.5/raw_tracks/"
)  # change path to where you want to store the EEP tracks


def __main__():

    initfeh_vals = [
        -4.0,
        -3.5,
        -3.0,
        -2.75,
        -2.5,
        -2.25,
        -2.0,
        -1.75,
        -1.5,
        -1.25,
        -1.0,
        -0.75,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
    ]
    alpha_vals = [-0.2, 0.0, 0.2, 0.4, 0.6]
    vvcrit_vals = [0.0, 0.4]

    for initfeh in initfeh_vals:
        for alpha in alpha_vals:
            for vvcrit in vvcrit_vals:
                if initfeh == 0.5 and alpha == 0.6:
                    # these parameter combinations are not available, so skip them
                    continue

                folder, url = _generate_MIST_EEP_url(initfeh, alpha, vvcrit)
                folder_path = EEP_PATH / folder
                if folder_path.is_dir():
                    print(f"{folder_path} is a directory.")
                else:
                    print(f"Downloading {url} to {folder_path}")
                    # dest_folder is what makes the skip-check above mean
                    # anything: without it every tarball landed in ./temp_files
                    # instead, folder_path never came into existence, and a
                    # re-run of the 186 GB workflow re-downloaded all of it.
                    download_tarfiles(url, dest_folder=folder_path)


if __name__ == "__main__":
    __main__()
