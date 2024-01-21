# ====================================================================
#  Importing the required python packages
# ====================================================================
import pandas as pd
import json
from legoai.modules.datatype_identification.preprocessing import remove_non_ascii
from legoai.modules.datatype_identification.preprocessing import data_standarization
import os
from tqdm import tqdm

# from core.logger import Logger

from legoai.core.model_configuration import MODEL_CONFIG


# Creating an logger object
# logger = Logger.getLogger(parent_folder_name="datatype_l1_identification",child_folder_name="feature")

# ====================================================================
# check_column_duplicates: 
#     - Read the column names and converts them to dataframe
#     - Clean and convert the column names into lower case
#     - Create an incremental values as cumulative count and replace the first value 
#     - Return the cleaned and updated column names
# Parameters: 
#     colNames - List of dataframe column names
# ====================================================================
    
def check_column_duplicates(colNames):
    colNames = [remove_non_ascii(col) for col in colNames]
    col_df = pd.DataFrame(colNames,columns = ['cols'])
    col_df['cleaned_cols'] = col_df['cols'].str.strip().str.lower()
    cum_cols_var = col_df['cleaned_cols'].str.cat(col_df.groupby(['cleaned_cols']).cumcount().add(1).astype(str),sep='##_')
    col_df['cols'] = col_df['cols'].str.cat(cum_cols_var.str.split('##').str[1].str.replace('_1', ''))
    colNames = col_df['cols'].tolist()
    return colNames

# ====================================================================
# input_file_transformation: 
#     - Read the csv files present in the inference file path
#     - Concatenate all the csv files into a combined df for inference
#     - Get the data, column name, table name and dataset name into dataframe
# Parameters: 
#     source_folder - Inference source file path
# ====================================================================

def input_file_transformation(source_folder: str) -> pd.DataFrame:
    
    ### data transformation for the required format
    df = pd.DataFrame()
    
    ### Iterating through source folder and read the file in encoding format
    #logger.debug('Reading label data %s', os.listdir(source_folder))
    for file_name in os.listdir(source_folder):
        
        ### Check if the file ends wit csv or not
        if not file_name.endswith('.csv'):
            #logger.debug(f'{file_name} not in csv format')
            continue

        ### read the input csv file format with datatype encoding and file encoding
        data = pd.read_csv(os.path.join(source_folder,file_name),encoding= MODEL_CONFIG['FILE_TYPE']['FILE_ENCODING'], dtype=MODEL_CONFIG['FILE_TYPE']['DTYPE_ENCODING'])
        #logger.debug('filename: %s, size: %s',file_name, data.shape)

        ### read the input csv file format with datatype encoding and file encoding
        data_values = []
        for idx, col_name in enumerate(data.columns):
               
            dataset_name = source_folder.split('/')[-1].strip()  ## Get the dataset name from source folder
            file_name = file_name.replace('.csv','').strip()  ## Get the file name from source folder
            
            ### Append the extracted data into temp dataframe
            data_values.append([dataset_name,file_name,col_name,dataset_name + '$$##$$' + file_name + '$$##$$' + col_name, 
                               data[col_name].tolist()])
        df = pd.concat([df,pd.DataFrame(data_values,columns=['dataset_name', 'table_name', 'column_name', 'master_id','values'])],ignore_index=True)
       
    ### Generate the master id and convert them to lower case
    df['master_id'] = df['master_id'].str.lower()
    df = df.reset_index(drop=True).reset_index()
    df = df.rename(columns={'index':'id'})
    
    return df
    
# ====================================================================
# source_file_conversion: 
#     - Iterate through each file present in the inference folder to get the file name
#     - Check if the file is excel/json/txt format read and write it as csv in processed folder
#     - If the file is only csv format, then we copy from inference to inference processed folder
# Parameters: 
#     folder_path - Inference source file path
# ====================================================================

def source_file_conversion(folder_path: str) -> str:
    t = tqdm(os.listdir(folder_path),desc="[*] preprocessing dataset...")
    ### Iterating through each file in the folder path
    for file_name in t:

        #logger.debug('filename: %s',file_name)

        ### Source and destination file path
        file_path = os.path.join(os.sep,folder_path,file_name)
        dest_path = file_path.rsplit('.',1)[0].replace('inference_repo','inference_repo_processed')+'.csv'
        dest_folder = os.path.split(dest_path)[0]
        
        ### Create the destination directory if not present
        if os.path.isdir(dest_folder):
            # print("[*] Directory exist." + dest_folder)
            pass
        else:
            #logger.debug("Directory does not exists. Creating new one. %s" + dest_folder)
            os.makedirs(dest_folder)

        ## Check if the file is csv file format and if yes copy
        if file_name.lower().endswith('.csv'):
            data = pd.read_csv(file_path)

        ## Check if the file is xlsx file format, then read the file
        elif file_name.lower().endswith('.xlsx') or file_name.endswith('.xls'):
            data = pd.read_excel(file_path)

        ## Check if the file is json file format, then read the file
        elif file_name.lower().endswith(('json','txt')):
            with open(file_path, 'r') as f:
                data_json = json.load(f)

            data = pd.DataFrame(data_json)

        else:
            continue    
        
        data = data.drop_duplicates().reset_index(drop=True)
        data.columns = check_column_duplicates(data.columns)
        data_std = data_standarization(data)
        # print(dest_path)
        data_std.to_csv(dest_path, index=False)

        t.set_description(f"[*] processed {file_name}",refresh=True)

        
    return dest_folder
