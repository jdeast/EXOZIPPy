import re
from typing import Literal

# -------------------------------------------------------------------
# MIST url and file name parsing
# -------------------------------------------------------------------

# compile pattern for MIST EEP tracks
_MISTv1_EEPTAR_FILENAME_RE = re.compile(r"MIST_v1\.2_feh_(?P<initfeh>[mp]\d+\.\d+)_afe_(?P<alpha>[mp]\d+\.\d+)_vvcrit(?P<vvcrit>\d+\.\d+)_EEPS\.txz")
_MISTv2_EEPTAR_FILENAME_RE = re.compile(r"MIST_v2\.5_feh_(?P<initfeh>[mp]\d+)_afe_(?P<alpha>[mp]\d+)_vvcrit(?P<vvcrit>\d+\.\d+)_EEPS\.txz")
MIST_EEP_FILENAME_RE = re.compile(r"(?P<initmass>\d+)M.track.eep")
MIST_BASE_URL = "https://mist.science"

_MIST_VERSIONS_PARAMS = {
    "1.2": {
        "tar_regex": _MISTv1_EEPTAR_FILENAME_RE,
        "tar_prefix": "MIST_v1.2_",
        "tar_ext": "_EEPS.txz",
        "download_url": MIST_BASE_URL + "/data/tarballs_v1.2/",
        "data_dir": ""
    },
    "2.5": {
        "tar_regex": _MISTv2_EEPTAR_FILENAME_RE,
        "tar_prefix": "MIST_v2.5_",
        "tar_ext": "_EEPS.txz",
        "download_url": MIST_BASE_URL + "/data/tarballs_v2.5/eeps/",
        "data_dir": "eeps/"
    },
}

_MP_TO_SIGN = {
    "p":  1.0,
    "m": -1.0
}


def _generate_MIST_EEP_url(initfeh: float, alpha: float, vvcrit: float, 
                            version: Literal["1.2", "2.5"] = "2.5") -> str:
    
    # initialize filename
    filename = _MIST_VERSIONS_PARAMS[version]["tar_prefix"] # example: "MIST_v2.5_"

    # add initfeh
    folder = "feh_"
    folder += "m" if initfeh < 0 else "p"
    folder += f"{abs(initfeh)*100:03.0f}_" if version == "2.5" else f"{abs(initfeh):0.2f}_"

    # add alpha
    folder += "afe_"
    folder += "m" if alpha < 0 else "p"
    folder += f"{abs(alpha)*10:.0f}_" if version == "2.5" else f"{abs(alpha):0.1f}_"

    # add vvcrit
    folder += "vvcrit"
    folder += f"{vvcrit:0.1f}"

    # add file ending
    filename += folder
    filename += _MIST_VERSIONS_PARAMS[version]["tar_ext"]

    # construct full url for downloading data
    url = _MIST_VERSIONS_PARAMS[version]["download_url"]

    return folder, url+filename


def _parse_initfeh_alpha_vvcrit_from_filename(filename: str, version: Literal["1.2", "2.5"] = "2.5"):

    m = _MIST_VERSIONS_PARAMS[version]["tar_regex"].match(filename)

    if not m:
        raise ValueError(f"Cannot parse parameters from filename: {filename}")

    sign_initfeh = _MP_TO_SIGN[m.group("initfeh")[0]]
    sign_alpha = _MP_TO_SIGN[m.group("alpha")[0]]

    initfeh = sign_initfeh*float(m.group("initfeh")[1:])
    alpha = sign_alpha*float(m.group("alpha")[1:])
    vvcrit = float(m.group("vvcrit"))

    if version == "2.5":
        initfeh *= 0.01
        alpha *= 0.1

    return(initfeh, alpha, vvcrit)