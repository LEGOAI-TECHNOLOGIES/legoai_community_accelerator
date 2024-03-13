# ====================================================================
#  Importing the required python packages
# ====================================================================
import traceback
import pandas as pd
from tqdm import tqdm

import os
import json

from legoai.modules.datatype_identification.preprocessing import remove_non_ascii
from legoai.modules.datatype_identification.preprocessing import data_standardization
from legoai.core.configuration import MODEL_CONFIG

# from core.logger import Logger

# Creating an logger object
# logger = Logger.getLogger(parent_folder_name="datatype_l1_identification",child_folder_name="feature")
def check_column_duplicates(colNames: list[str]) -> list[str]:
    """
    - Read the column names and converts them to dataframe
    - Clean and convert the column names into lower case
    - Create an incremental values as cumulative count and replace the first value
    - Return the cleaned and updated column names

    Parameters
    ----------
    colNames (list): list of column names

    Returns
    -------
    list of cleaned column names
    """
    colNames = [remove_non_ascii(col) for col in colNames]
    col_df = pd.DataFrame(colNames, columns=['cols'])
    col_df['cleaned_cols'] = col_df['cols'].str.strip().str.lower()
    cum_cols_var = col_df['cleaned_cols'].str.cat(col_df.groupby(['cleaned_cols']).cumcount().add(1).astype(str),
                                                  sep='##_')
    col_df['cols'] = col_df['cols'].str.cat(cum_cols_var.str.split('##').str[1].str.replace('_1', ''))
    colNames = col_df['cols'].tolist()
    return colNames


def input_file_transformation(source_folder: str) -> pd.DataFrame:
    """
    - Read the csv files present in the inference file path
    - Concatenate all the csv files into a combined df for inference
    - Get the data, column name, table name and dataset name into dataframe

    Parameters
    ----------
    source_folder (str): path to the pre-processed files

    Returns
    -------
    dataframe with all relevant columns and values
    """

    ### data transformation for the required format
    df = pd.DataFrame()

    ### Iterating through source folder and read the file in encoding format
    # logger.debug('Reading label data %s', os.listdir(source_folder))
    for file_name in os.listdir(source_folder):

        ### Check if the file ends wit csv or not
        if not file_name.endswith('.csv'):
            # logger.debug(f'{file_name} not in csv format')
            continue

        ### read the input csv file format with datatype encoding and file encoding
        data = pd.read_csv(os.path.join(source_folder, file_name), encoding=MODEL_CONFIG['FILE_TYPE']['FILE_ENCODING'],
                           dtype=MODEL_CONFIG['FILE_TYPE']['DTYPE_ENCODING'])
        # logger.debug('filename: %s, size: %s',file_name, data.shape)

        ### read the input csv file format with datatype encoding and file encoding
        data_values = []
        for idx, col_name in enumerate(data.columns):
            dataset_name = os.path.split(source_folder)[-1].strip()
            # dataset_name = source_folder.split('/')[-1].strip()  ## Get the dataset name from source folder
            file_name = file_name.replace('.csv', '').strip()  ## Get the file name from source folder

            ### Append the extracted data into temp dataframe
            data_values.append(
                [dataset_name, file_name, col_name, dataset_name + '$$##$$' + file_name + '$$##$$' + col_name,
                 data[col_name].tolist()])
        df = pd.concat([df, pd.DataFrame(data_values,
                                         columns=['dataset_name', 'table_name', 'column_name', 'master_id', 'values'])],
                       ignore_index=True)

    ### Generate the master id and convert them to lower case
    df['master_id'] = df['master_id'].str.lower()
    df = df.reset_index(drop=True).reset_index()
    df = df.rename(columns={'index': 'id'})

    return df


def source_file_conversion(folder_path: str, output_path: str) -> str:
    """
    - Iterate through each file present in the inference folder to get the file name
    - Check if the file is excel/json/txt format read and write it as csv in processed folder
    - If the file is only csv format, then we copy from inference to inference processed folder

    Parameters
    ----------
    folder_path (str): path where the data for inference is located

    Returns
    -------
    final path of pre-processed files
    """

    t = tqdm(os.listdir(folder_path), desc="[*] preprocessing dataset...")
    dest_folder = os.path.join(output_path,
                               os.path.split(folder_path)[1] + "_processed")

    ### Iterating through each file in the folder path
    for file_name in t:
        # logger.debug('filename: %s',file_name)

        ### Source and destination file path
        file_path = os.path.join(folder_path, file_name)
        # print(file_path)

        dest_path = os.path.join(dest_folder, file_name)

        ### Create the destination directory if not present
        if os.path.isdir(dest_folder):
            # print("[*] Directory exist." + dest_folder)
            pass
        else:
            # logger.debug("Directory does not exists. Creating new one. %s" + dest_folder)
            os.makedirs(dest_folder)

        ## Check if the file is csv file format and if yes copy
        if file_name.lower().endswith('.csv'):
            data = pd.read_csv(file_path)

        ## Check if the file is xlsx file format, then read the file
        elif file_name.lower().endswith('.xlsx') or file_name.endswith('.xls'):
            data = pd.read_excel(file_path)

        ## Check if the file is json file format, then read the file
        elif file_name.lower().endswith(('json', 'txt')):
            with open(file_path, 'r') as f:
                data_json = json.load(f)

            data = pd.DataFrame(data_json)

        else:
            continue

        data = data.drop_duplicates().reset_index(drop=True)
        data.columns = check_column_duplicates(data.columns)
        data_std = data_standardization(data)
        # print(dest_path)
        data_std.to_csv(dest_path, index=False)

        t.set_description(f"[*] processed {file_name}", refresh=True)



    return dest_folder


# For training purposes checking meta information about the table
def meta_information_check(data_df: pd.DataFrame, filename: str, reponame: str) -> pd.DataFrame:
    """
    - Dataframe with columns and values is passed as parameter to function
    - Check if the unique "id" column is present or not. If not present then create index as the "id" column
    - Check if the table name or column name is present, if not derive from file name or assign it as empty string
    - Create master id from the repo name, table name and column name

    Parameters
    ----------
    data_df (pd.DataFrame): dataframe with values and related columns
    filename (str): input filename
    reponame (str): input repo / dataset name

    Returns
    -------
    Dataframe with relevant and required columns along with additional metadata info
    """

    ### ID creation for dataframe if not present
    if 'id' in data_df.columns:

        if data_df['id'].nunique() != data_df.shape[0]:
            data_df['id'] = [i for i in range(len(data_df))]

            ### If column id is present, then we use column id as id
    elif 'column_id' in data_df.columns:
        data_df['id'] = data_df['column_id']
    else:
        data_df = data_df.reset_index(drop=True).reset_index()
        data_df = data_df.rename(columns={'index': 'id'})

    #### Mandatory column checks for column name
    if 'column_name' not in data_df.columns:

        if 'type' in data_df.columns:  ### Assigning "type" as column name if they are not present
            data_df['column_name'] = data_df['type']
        else:
            data_df['column_name'] = ''  ### If column name is not present, then we assign it as empty

    #### Mandatory column checks for table name            
    if 'table_name' not in data_df.columns:
        data_df['table_name'] = filename.rsplit('.', 1)[
            0]  ### If table name is not present, then extract from file name

    ### Renaming certain column values to values
    data_df = data_df.rename(columns={'column_values': 'values', 'value': 'values'})

    ### Convert the datatype to list format
    data_df['values'] = data_df.apply(lambda x: list(x['values']), axis=1)
    # print('Unique Type of values:', list(set([type(val) for val in data_df['values'].tolist()])))

    ### Check if the values are of list types
    assert list(set([type(val) for val in data_df['values'].tolist()]))[0] == list



    ### If dataset name given set it as repo name
    if 'dataset_name' in data_df.columns:
        data_df['repo_name'] = data_df['dataset_name']
    else:
        data_df['repo_name'] = reponame

    ### For Specific repo name, we are using dataset name as repo name
    # if reponame == 'db_extract':
    #     data_df['repo_name'] = data_df['dataset_name']
    # else:
    #     data_df['repo_name'] = reponame

    ### Creating the master id from repo name, table name and column name and convert to lower type
    data_df['master_id'] = data_df.apply(
        lambda x: x['repo_name'] + '$$##$$' + x['table_name'] + '$$##$$' + x['column_name'], axis=1)
    data_df['master_id'] = data_df['master_id'].str.lower()

    ### Check if the master id is unique or not
    assert data_df['master_id'].nunique() == data_df.shape[0]

    ### Convert id to string
    # data_df.id = data_df.id.astype(str)

    return data_df


def data_conversion(filename: str, filepath: str) -> pd.DataFrame:
    """
     - Pass the filename and filepath as input parameter to the function.
     - Extract the file extension from the filename
     - Based on the extension, read the respective file formats
     - Convert the file into dataframe format and return the dataframes

    Parameters
    ----------
    filename(str): Input filename
    filepath (str): Input file path

    Returns
    -------
    Dataframe with relevant and required columns
    """

    ## Identify the file extension
    file_extension = filename.rsplit('.', 1)[1]

    ## Read the data based on extension

    ### File format in json or txt format
    if file_extension in ('json', 'txt'):
        with open(filepath, 'r') as f:
            data_json = json.load(f)

        data = pd.DataFrame(data_json)

    ### File format in parquet format
    elif file_extension == 'parquet':
        data = pd.read_parquet(filepath)

    ### File format in csv format
    elif file_extension == 'csv':
        data = pd.read_csv(filepath)

    ### else return empty dataframe if they are in neither format
    else:
        data = pd.DataFrame(columns=['id', 'table_name', 'column_name', 'values'])

    return data


def extract_file_meta_info(dataset_path: str) -> pd.DataFrame:
    """
    - Iterating through the Raw data path for and reading the filenames present
    - Checking if the files are of json type and split the path to get the folder/repo name, file name and file path for each
    - Create the list of file related info into a dataframe

    Parameters
    ----------
    dataset_path(str): repo or folder path

    Returns
    -------
    Dataframe with relevant and required columns along with additional metadata info
    """
    print("[*] Extracting files meta information ...")
    data_content = []
    # data_path = os.path.join(PATH_CONFIG['CONTAINER_PATH'], dataset_path)
    data_path = dataset_path
    try:
        for path, subdirs, files in os.walk(data_path):
            for name in files:
                data_content.append(os.path.join(path, name))
        _meta_info = []
        for cont in data_content:
            file_meta_informations = cont.split(os.sep)
            source, location, reponame, filename = file_meta_informations[1], file_meta_informations[-3], \
            file_meta_informations[-2], file_meta_informations[-1]
            _meta_info.append([source, location, reponame, filename, cont])

        _meta_df = pd.DataFrame(_meta_info, columns=['source', 'location', 'reponame', 'filename', 'filepath'])
        return _meta_df

    except Exception as e:
        print(traceback.format_exc())
        print(e)


def generate_id(id: str, name: str) -> str:
    """
     Generate ID for training data
    - Based on the type of repo passed as parameter then we create the respective id
    - Return the final ID based on the repo as column id

    Parameters
    ----------
    id (str):
    name (str): dataset name

    Returns
    -------
    id for training data
    """

    ### Check if the data is real world data features, then retain the ID
    if name == 'real_world_dataset':
        if id.startswith('ID_'):
            return id
        else:
            return 'SH_' + str(id)
    elif name == 'web_crawl_dataset':
        return 'WDC_' + str(id)
    else:
        return id
