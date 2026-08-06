import os
import requests
import tarfile
from pathlib import Path

from .parse_MIST_EEP_filenames import _generate_MIST_EEP_url


# -------------------------------------------------------------------
# Download Files
# -------------------------------------------------------------------

def download_tarfiles(url, dest_folder="temp_files"):
    os.makedirs(dest_folder, exist_ok=True)
    tar_path = os.path.join(dest_folder, "archive.tar.gz")

    # 1. Download the file safely in chunks
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(tar_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024): #chunk_size=8192
            f.write(chunk)

    # 2. Extract (untar) the files
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=dest_folder)


# -------------------------------------------------------------------
# Download All MISTv2.5 EEP Tracks
# -------------------------------------------------------------------

EEP_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/raw_tracks/") # change path to where you want to store the EEP tracks

def __main__():

    initfeh_vals = [ -4.0,  -3.5,  -3.0, -2.75,  
           -2.5, -2.25,  -2.0, -1.75,  
           -1.5, -1.25, -1.0, -0.75,  
           -0.5, -0.25,   0.0,  0.25,   0.5]
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
                    print(f"Downloading {url} to {folder}")
                    download_tarfiles(url)

if __name__ == "__main__":
    __main__()