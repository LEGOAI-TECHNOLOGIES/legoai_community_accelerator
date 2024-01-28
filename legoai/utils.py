import openai

import os
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import sys

from legoai.core.configuration import PATH_CONFIG

DEFAULT_DATASET = "ecommerce_data"

def download_default_dataset():
    '''
        Downloads the default dataset "ecommerce" from the path configured URL
        Parameters:
            None

        Returns:
            DEFAULT_DATASET (str): name of the default dataset
    '''
    # path to the default dataset
    inferece_data_path = os.path.join(PATH_CONFIG["CONTAINER_PATH"],PATH_CONFIG["INF_DATA_PATH"],"inference_repo",DEFAULT_DATASET)
    
    download = True
    # create the inference input dir if not exists
    try:
        os.makedirs(inferece_data_path)
    except FileExistsError:
        # if exists check if the directory has any file then donot download.
        if len(os.listdir(inferece_data_path)) > 0:
            print("[*] dataset already downloaded")
            download = False

    if download:
        with urlopen(PATH_CONFIG["DATASET_PATH_URL"]+DEFAULT_DATASET+".zip") as zipresp:
            print(f"[*] downloading and extracting {DEFAULT_DATASET}")
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(os.path.split(inferece_data_path)[0])
    
    return DEFAULT_DATASET


def check_openai_key(key:str):
    key_clean = key.strip().replace(" ","") if key is not None else ""
    if len(key_clean) == 0:
        print("[!] openai api key not provided")
        sys.exit(-1)
    else:
        try:
            openai.api_key = key
            openai.Model.list()
        except openai.error.AuthenticationError as e:
            print("[!] provide valid openai key")
            sys.exit(-1)
