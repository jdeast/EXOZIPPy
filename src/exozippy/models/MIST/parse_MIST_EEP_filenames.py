import re
from typing import Literal

# -------------------------------------------------------------------
# MIST url and file name parsing
# -------------------------------------------------------------------

# single pattern for MIST EEP tar files, folder names, and parquet files:
# - optional "MIST[_]v<version>_" prefix matches tar filenames
# - optional "_EEPS.txz" suffix matches tar filenames
# - trailing content (e.g. "/" for folders, ".parquet" for files) is left
#   unmatched since match() doesn't require matching to the end of the string
_MIST_EEP_NAME_RE = re.compile(
    r"(?:MIST_?v(?P<version>[\d.]+)_)?"
    r"feh_(?P<initfeh>[mp]\d+(?:\.\d+)?)"
    r"_afe_(?P<alpha>[mp]\d+(?:\.\d+)?)"
    r"_vvcrit(?P<vvcrit>\d+\.\d+)"
    r"(?:_EEPS\.txz)?"
)

MIST_EEP_FILENAME_RE = re.compile(r"(?P<initmass>\d+)M.track.eep")
MIST_BASE_URL = "https://mist.science"

_MIST_VERSIONS_PARAMS = {
    "1.2": {
        "download_url": MIST_BASE_URL + "/data/tarballs_v1.2/",
        "data_dir": "",
        "folder_regex": _MIST_EEP_NAME_RE,
    },
    "2.5": {
        "download_url": MIST_BASE_URL + "/data/tarballs_v2.5/eeps/",
        "data_dir": "eeps/",
        "folder_regex": _MIST_EEP_NAME_RE,
    },
}

_MP_TO_SIGN = {"p": 1.0, "m": -1.0}


def _generate_MIST_EEP_url(
    initfeh: float,
    alpha: float,
    vvcrit: float,
    version: Literal["1.2", "2.5"] = "2.5",
) -> str:

    # initialize filename
    filename = f"MIST_v{version}_"  # example: "MIST_v2.5_"

    # add initfeh
    folder = "feh_"
    folder += "m" if initfeh < 0 else "p"
    folder += (
        f"{abs(initfeh) * 100:03.0f}_"
        if version == "2.5"
        else f"{abs(initfeh):0.2f}_"
    )

    # add alpha
    folder += "afe_"
    folder += "m" if alpha < 0 else "p"
    folder += (
        f"{abs(alpha) * 10:.0f}_"
        if version == "2.5"
        else f"{abs(alpha):0.1f}_"
    )

    # add vvcrit
    folder += "vvcrit"
    folder += f"{vvcrit:0.1f}"

    # add file ending
    filename += folder
    filename += "_EEPS.txz"

    # construct full url for downloading data
    url = _MIST_VERSIONS_PARAMS[version]["download_url"]

    return folder, url + filename


def _parse_initfeh_alpha_vvcrit_from_name(
    name: str, version: Literal["1.2", "2.5"] = "2.5"
):

    if version not in _MIST_VERSIONS_PARAMS:
        raise ValueError(f"Unsupported MIST version: {version}")

    # check if the name matches the expected pattern
    m = _MIST_EEP_NAME_RE.match(name)

    if not m:
        raise ValueError(f"Cannot parse parameters from: {name}")

    sign_initfeh = _MP_TO_SIGN[m.group("initfeh")[0]]
    sign_alpha = _MP_TO_SIGN[m.group("alpha")[0]]

    initfeh = sign_initfeh * float(m.group("initfeh")[1:])
    alpha = sign_alpha * float(m.group("alpha")[1:])
    vvcrit = float(m.group("vvcrit"))

    if version == "2.5":
        initfeh *= 0.01
        alpha *= 0.1

    return (initfeh, alpha, vvcrit)


def _parse_initmass_from_filename(filename: str) -> float:

    m = MIST_EEP_FILENAME_RE.match(filename)

    if not m:
        raise ValueError(
            f"Cannot parse initial mass from filename: {filename}"
        )

    initmass = float(m.group("initmass"))
    initmass *= 0.01

    return initmass
