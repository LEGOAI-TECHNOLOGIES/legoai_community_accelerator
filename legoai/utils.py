from typing import List

import openai
import pandas as pd

import os
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import sys
from datetime import datetime
import json


from legoai.core.configuration import PATH_CONFIG

DEFAULT_DATASET = "ecommerce_data"

VALID_DATASET_EXTENSIONS = ['csv', 'json', 'txt', 'xslx']

DATASET_PATH_URL = "http://localhost:8000/Lego_AI/inference/"

def check_dataset_path(*args):
    """
    - Checks if the given path has valid files or not
    Parameters
    ----------
    args: list of paths to check for valid files

    Returns
    -------

    """
    for dataset_path in args:
        if dataset_path is None or not os.path.exists(dataset_path):
            print(f"\n[!] Given path {dataset_path} not valid")
            sys.exit(-1)
        elif not len(os.listdir(dataset_path)) > 0:
            raise FileNotFoundError(
                f"\n[!] Given path {dataset_path} doesn't hold any files for processing"
            )
        else:
            for path, subdir, files in os.walk(dataset_path):
                for file in files:
                    extension = file.split(".")[-1]
                    if extension not in VALID_DATASET_EXTENSIONS:
                        raise Exception(
                            f"\n[!] {os.path.join(path, file)} is not a valid file ... must be one of {VALID_DATASET_EXTENSIONS}"
                        )




def download_default_dataset():
    """
    - Downloads the default dataset "ecommerce" from the path configured URL
    Parameters
    ---------

    Returns
    -------
    path to the default example inference data
    """
    # path to default dataset
    inference_data_path = os.path.join(f"default_data_{datetime.now().strftime('%d%m%Y')}",
                                       "inference",
                                       "inference_repo",
                                       DEFAULT_DATASET)

    # path to the default dataset
    # inference_data_path = os.path.join(PATH_CONFIG["CONTAINER_PATH"],PATH_CONFIG["INF_DATA_PATH"],"inference_repo",DEFAULT_DATASET)

    download = True
    # create the inference input dir if not exists
    try:
        os.makedirs(inference_data_path)
    except FileExistsError:
        # the directory has any file then don't download.
        if len(os.listdir(inference_data_path)) > 0:
            print("[*] dataset already downloaded")
            download = False

    if download:
        with urlopen(DATASET_PATH_URL + DEFAULT_DATASET + ".zip") as zipresp:
            print(f"[*] downloading and extracting {DEFAULT_DATASET}")
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(os.path.split(inference_data_path)[0])

    return inference_data_path


def check_openai_key(key: str):
    """
    - Checks provided openai key ( requires internet )
    Parameters
    ----------
    key (str): api key

    Returns
    -------

    """
    key_clean = key.strip().replace(" ", "") if key is not None else ""
    if len(key_clean) == 0:
        raise ValueError(
            "[!] openai api key not provided"
        )
    else:
        try:
            openai.api_key = key
            openai.Model.list()
        except openai.error.AuthenticationError:
            print("[!] provide valid openai key")
            sys.exit(-1)
        except openai.error.APIConnectionError:
            print("[!] internet connection not available for l3 model ...")
            sys.exit(-1)


REQUIRED_GT_COLUMNS = {"master_id","datatype"}
def combine_gt_file(path:str) -> pd.DataFrame:
    """
    Combines all the ground truth files for training
    Parameters
    ----------
    path (str): directory path to ground truth data

    Returns
    -------
        pd.DataFrame: combined ground truth dataframe
    """
    gt_df = pd.DataFrame()
    for path, subdir, files in os.walk(path):
        for file in files:
            df = pd.DataFrame()
            extension = file.split(".")[-1]
            file_path = os.path.join(path,file)
            if extension == "csv":
                df = pd.read_csv(file_path)
            elif extension in ('json', 'txt'):
                with open(os.path.join(file_path), 'r') as f:
                    data_json = json.load(f)
                    df = pd.DataFrame(data_json)
            elif extension == "xslx":
                df = pd.read_excel(file_path)

            df.columns = map(str.lower, df.columns)
            if REQUIRED_GT_COLUMNS.issubset(df.columns.tolist()):
                gt_df = pd.concat([gt_df,df[['master_id','datatype']]])
            else:
                raise Exception("[!] 'master_id' and 'datatype' columns not present in ground truth file...")

    return gt_df





