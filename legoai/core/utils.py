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
from pathlib import Path
import itertools
from multiprocessing.pool import ThreadPool


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
            raise ValueError(f"\n[!] Given path {dataset_path} not valid")

        elif os.path.isdir(dataset_path) and not len(os.listdir(dataset_path)) > 0:
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
    key (str): open ai api key

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
            openai.models.list()
        except openai.AuthenticationError:
            print("[!] provide valid openai key")
            sys.exit(-1)
        except openai.APIConnectionError:
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
    gt_files = []

    if os.path.isdir(path):
        for gt_path, subdirs, files in os.walk(path):
            for name in files:
                gt_files.append(os.path.join(gt_path, name))
    elif os.path.isfile(path):
        gt_files.append(path)

    for file_path in gt_files:
        df = pd.DataFrame()
        extension = file_path.split(os.sep)[-1].split(".")[-1]
        # file_path = os.path.join(path,file)

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
            gt_df = pd.concat([gt_df,df[list(REQUIRED_GT_COLUMNS)]])
        else:
            raise Exception(f"[!] {REQUIRED_GT_COLUMNS} columns not present in ground truth file...")

    return gt_df

def load_file(path):
    """
    Load files with certain extensions as pandas DataFrame
    Parameters
    ----------
    path (str): full path to the file

    Returns
    -------
    pd.DataFrame: file read as pandas DataFrame
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{path} doesn't exists")

    if not path.is_file() and path.is_dir():
        raise ValueError(f"{path} must be a file not a directory")
    else:
        df = pd.DataFrame()
        if path.name.endswith(".xlsx"):
            df = pd.read_excel(path)
        elif path.name.endswith(".csv"):
            df = pd.read_csv(path, low_memory=False)
        elif path.name.endswith(".json") or path.name.endswith(".txt"):
            with path.open('r') as file:
                data = json.load(file)
                df = pd.DataFrame(data)
        else:
            raise ValueError(f"{path} is invalid file format")

        assert df.shape != (0, 0), f"{path} file has no data... check the file"
        return df


def parallel_json_converter(file_path):
    """
    Helper funtion for preparing training json file

    Parameters:
    ----------
    file_path (str): full path to the training file

    Returns:
    --------
    List[dict]: List of dictionary training data


    """
    df = load_file(file_path)

    # fill nan values by backfill method
    df.bfill(inplace=True)
    # drop if there is null values even now
    df.dropna(inplace=True, axis=1)

    json_data = []
    path = Path(file_path)
    repo_name = path.parent.name
    table_name = path.name.split(".")[0]

    for column_name in df.columns:
        json_data.append(
            {
                'master_id': repo_name.lower() + "$$##$$" + table_name.lower() + "$$##$$" + column_name.lower(),
                'dataset_name': repo_name,
                'table_name': table_name,
                'column_name': column_name,
                'column_values': df[column_name].values.tolist()
            }
        )
    return json_data


def prepare_di_training_file(path):
    """
    Prepare Training data from list of training files

    Parameters
    ----------
    path (str): path to the directory that contains training files.

    Returns
    -------
    List[dict]: list of dictionary containing training data

    """
    training_files = []
    # get all the training files
    for path, subdirs, files in os.walk(path):
        for file in files:
            training_files.append(os.path.join(path, file))

    training_json = []
    with ThreadPool() as pool:
        training_json = pool.map(parallel_json_converter, training_files)

    return list(itertools.chain(*training_json))

def prepare_di_ground_truth(json_training):
    """
    Prepare Excel ground truth file for manual labelling


    """
    data = {'master_id': [], 'column_values': []}

    for training in json_training:
        data['master_id'].append(training['master_id'])
        only_display_upto = min(len(training['column_values']), 10)
        data['column_values'].append(training['column_values'][:only_display_upto])

    df = pd.DataFrame(data)
    df['datatype'] = ''
    assert df.shape[0] == len(json_training),"Training & Ground truth don't have same number of values"
    return df


REQUIRED_GT_COLUMNS = {'master_id', 'datatype'}
REQUIRED_TRAINING_COLUMNS = {'master_id', 'dataset_name', 'column_values', 'column_name', 'table_name'}
ALLOWED_DATATYPES = {'integer', 'float', 'range_type', 'date & time', 'open_ended_text', 'close_ended_text','alphanumeric','others'}
tick = u'\u2713'


def check_columns(df, columns, path):
    assert columns.issubset(set(df.columns)), f"columns {columns} not in {path}... required for working"


def check_null_values(df, column, path):
    assert df[column].isna().sum() == 0, f"{column} in {path} has null values"


def check_empty_null_values(data, master_id, column, path):
    error_msg = f"{column} has empty or null value with master_id: {master_id} at {path}"

    if isinstance(data, list):
        assert any(pd.isna(data)) == False, error_msg
        assert len(data) != 0, error_msg
    else:
        assert not pd.isna(data), error_msg
        if isinstance(data, str):
            assert not str(data).strip().__eq__(''), error_msg


def check_di_gt_labels(datatype, master_id, path):
    assert isinstance(datatype, str), f"datatype must be of string type at master_id: {master_id} at {path}"
    assert not str(datatype).strip().__eq__(''), f"master_id: {master_id} has empty or null datatype at {path}"
    assert str(
        datatype) in ALLOWED_DATATYPES, f"{datatype} is not one of {ALLOWED_DATATYPES} for master_id: {master_id} at {path}"


def precheck_di_training_file(path, required_columns, _display_message, file_type='training', **kwargs):
    """
    - Check training & gt file for any inconsitencies.
    - Checks for null , missing values, missing master_id,
    - Checks for missing datatype column in gt & validates against ALLOWED_DATATYPES

    Parameters
    ----------
    path (str): can be path to training or gt file
    required_columns (List): list of columns to check in training & gt.
    _display_message (boolean): flag to whether display print statements or not in the function.
    file_type (str): default 'training' , denotes file type training or gt.

    Returns 
    -------
    pd.DataFrame: gt or training file as dataframe
    """
    df = load_file(path)

    check_columns(df, required_columns, path)
    if _display_message:
        print(f"[{tick}] Required columns present in {path}")

    assert df['master_id'].isna().sum() == 0, f"{path} has empty or null master_id"
    if _display_message:
        print(f"[{tick}] No empty or Null master_id present in {path}")

    required_columns = required_columns - {'master_id'}

    for column in required_columns:
        df[[column, 'master_id']].apply(lambda x: check_empty_null_values(x[column], x['master_id'], column, path),
                                        axis=1)
    if _display_message:
        print(f"[{tick}] No null values present in {path}")

    if file_type == 'gt':
        df[['datatype', 'master_id']].apply(lambda x: check_di_gt_labels(x['datatype'], x['master_id'], path), axis=1)
        if _display_message:
            print(f"[{tick}] Ground truth labels checked for null , empty values & allowed data types")

    return df


def add_data_validation_excel_gt(gt_df:pd.DataFrame,save_path:str):
    """
    - Add excel data validation for allowing to fill the datatype column with only ALLOWED_DATATYPES
    - And finally save the excel gt file
    
    Parameters
    ----------
    gt_df (pd.DataFrame): pandas dataframe with the ground truth.
    save_path (str): full excel path to save the gt file.

    Returns
    -------
    """

    with pd.ExcelWriter(save_path,engine='xlsxwriter') as writer:
        gt_df.to_excel(excel_writer=writer,sheet_name="Sheet1",index = False)

        worksheet = writer.sheets['Sheet1']
        validation_column_start = 2 # top header would not have validation so start from C2
        validation_column_end = gt_df.shape[0] + 1 # plus one for excluding top header and starting from second row
        worksheet.data_validation(
          f'C{validation_column_start}:C{validation_column_end}',
            {
                'validate':'list',
                'source': list(ALLOWED_DATATYPES),
                'input_title': 'Choose a datatype',
                'input_message': 'Select true datatype for the column_values from the list',

            }
        )



