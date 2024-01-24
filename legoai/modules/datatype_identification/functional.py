# ====================================================================
#  Importing the required python packages
# ====================================================================
import os
import random
from collections import OrderedDict
from datetime import datetime
import pandas as pd
from symspellpy.symspellpy import SymSpell
import multiprocessing


# from core.logger import Logger
from legoai.core.path_configuration import PATH_CONFIG
from legoai.core.configuration import MODEL_CONFIG

# ====================================================================
#  Importing the required custom module packages
# ====================================================================
from functional import pseq
from legoai.modules.datatype_identification.custom_features import extract_additional_feats
from legoai.modules.datatype_identification.preprocessing import normalise_string_whitespace, special_token_repl, additional_processing, \
    remove_table_column_name

# Creating an logger object
# logger = Logger.getLogger(
#     parent_folder_name = "datatype_l1_identification",child_folder_name="feature"
# )

# ====================================================================
# Set max_dictionary_edit_distance to 0 to avoid spelling correction
# ====================================================================
model_dep_path = os.path.join(PATH_CONFIG["CONTAINER_PATH"],PATH_CONFIG["MODEL_DEP_PATH"],"datatype_l1_identification")

sym_spell = SymSpell(max_dictionary_edit_distance=MODEL_CONFIG['THRESHOLD']['SYMSPELL_EDIT_DIST'], prefix_length=MODEL_CONFIG['THRESHOLD']['PARTITION_SIZE'])
dictionary_path = os.path.join(model_dep_path,'en-80k.txt')

# ====================================================================
# term_index is the column of the term and count_index is the column of the term frequency
# ====================================================================
sym_spell.load_dictionary(dictionary_path, term_index=MODEL_CONFIG['THRESHOLD']['SYMSPELL_TERM_INDEX'], count_index=MODEL_CONFIG['THRESHOLD']['SYMSPELL_COUNT_INDEX'])

# ====================================================================
# CPU Core count for multiprocessing and size of partition size
# ====================================================================
core_count = multiprocessing.cpu_count()  
size = MODEL_CONFIG['THRESHOLD']['PARTITION_SIZE']



# ====================================================================
# extract_features: 
#     - Extract the required metadata information from values list before feature creation
#     - Clean the table and column name for the embedding creation
#     - Additional preprocessing on the data and passed to the feature creation
#     - Created features one by one from characters, words, and embedding based features
#     - Created the none related features followed by metadata features added
# Parameters: 
#     col_values - List of data for feature creation
# Returns:
#     Returns the features dictionary for the input data
# ====================================================================

def extract_features(col_values: list) -> OrderedDict:
        
    ### Extract the table data column from column values
    master_id = col_values[0]
    id = col_values[1]
    dataset_name = col_values[2]
    table_name = col_values[3]
    column_name = col_values[4]
    col_values = col_values[5:]

    ### Cleaning the table name for special token replacement and segmenting the compound words
    table_name_clean = special_token_repl(table_name, suffix='_table_name')
    table_name_clean = sym_spell.word_segmentation(table_name_clean).corrected_string

    ### Cleaning the column name for special token replacement and segmenting the compound words
    column_name_clean = special_token_repl(column_name, suffix='_column_name')
    column_name_clean = sym_spell.word_segmentation(column_name_clean).corrected_string

    ### Number of samples used for feature creation and total samples present in data
    n_samples = MODEL_CONFIG['THRESHOLD']['DATA_VALUES_LIMIT']
    n_values = len(col_values)
    features = OrderedDict()

    ### Additional processing on the col values with nan and lower case converted
    # logger.info('Custom Preprocessing started:%s', datetime.now())
    cleaned_population_nan = pseq(map(additional_processing, col_values), processes=core_count, partition_size=size)
    cleaned_population_nan = list(cleaned_population_nan)

    # logger.info('Custom preprocessing completed:%s', datetime.now())

    ### Based on the number of values, either sample the data or take the entire values
    if n_samples < n_values:
        random.seed(13)
        cleaned_sample_nan = random.sample(cleaned_population_nan, k=n_samples)
    else:
        n_samples = n_values
        cleaned_sample_nan = cleaned_population_nan

    ### Additional processing on the col values without nan and lower case converted
    cleaned_sample_wo_nan = [val for val in cleaned_sample_nan if len(val) > 0]
    cleaned_sample_wo_nan_uncased = [val.lower() for val in cleaned_sample_wo_nan]

    ### Extracting the additional statistical features ( l1 model specific )
    # logger.info('*' * 100)
    extract_additional_feats(cleaned_sample_nan,cleaned_sample_wo_nan_uncased,features)
    # logger.info('*' * 100)
    
    ### Adding the metadata specific information of the features
    features['master_id'] = master_id
    features['id'] = id
    features['dataset_name'] = dataset_name
    features['table_name'] = table_name
    features['column_name'] = column_name
    features['table_name_clean'] = table_name_clean
    features['column_name_clean'] = column_name_clean

    # logger.info('Completed...')
    # logger.info('#' * 100)

    return features


# ====================================================================
# extract_features_to_csv: 
#     - Remove the table and column from the data points for feature creation
#     - 
# Parameters: 
#     parquet_df - Dataframe with the data for feature creation
# ====================================================================

def extract_features_to_csv(parquet_df: pd.DataFrame) -> pd.DataFrame:
    start = datetime.now()

    print("[*] Feature Creation Started...")
    features_df = pd.DataFrame()
    
    ### Remove the table and column from the data points for feature creation
    parquet_df['clean_values'] = parquet_df.apply(
        lambda x: remove_table_column_name(x['values'], x['dataset_name'], x['table_name'], x['column_name']), axis=1)

    ### Converting the metadata into string format
    parquet_df['master_id'] = parquet_df['master_id'].astype(str)
    parquet_df['id'] = parquet_df['id'].astype(str)
    parquet_df['dataset_name'] = parquet_df['dataset_name'].astype(str)
    parquet_df['table_name'] = parquet_df['table_name'].astype(str)
    parquet_df['column_name'] = parquet_df['column_name'].astype(str)

    ### Converting the values data into list format
    master_id = parquet_df['master_id'].values.tolist()
    id = parquet_df['id'].values.tolist()
    dataset_name = parquet_df['dataset_name'].values.tolist()
    table_name = parquet_df['table_name'].values.tolist()
    column_name = parquet_df['column_name'].values.tolist()
    data_values = parquet_df['clean_values'].values.tolist()

    ### Combining the metadata + values data
    parquet_values = [
        [master_id[val]] + [id[val]] + [dataset_name[val]] + [table_name[val]] + [column_name[val]] + list(
            data_values[val]) for val in range(len(parquet_df))]

    ### preprocessing the metadata + values data with whitespace removal
    normalized_list = pseq(map(normalise_string_whitespace, parquet_values), processes=core_count, partition_size=size)
    features_dict = pseq(map(extract_features, normalized_list), processes=core_count, partition_size=size)
    features_df = pd.DataFrame.from_dict(features_dict)

    ### Get the total number of null features and replace them with 0
    # logger.info('Null Features column: %s', [col for col in features_df.columns if sum(pd.isnull(features_df[col])) > 0])
    features_df = features_df.fillna(0)
    
    ### Get the feature execution stats
    features_df['start_time'] = start
    features_df['end_time'] = datetime.now()
    features_df['execution_time'] = datetime.now() - start

    print(f"[*] Feature Creation Finished. Processed {len(parquet_df)} rows in {(datetime.now() - start)}")
    
    return features_df