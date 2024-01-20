
from legoai.modules.datatype_identification.utils import source_file_conversion,input_file_transformation
from legoai.modules.datatype_identification.functional import extract_features_to_csv
from legoai.core.path_configuration import PATH_CONFIG
from legoai.modules.datatype_identification.l1_model import L1Model
from legoai.modules.datatype_identification.l3_model import L3Model

import pandas as pd
import os , sys
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class DataTypeIdentificationPipeline():


    def _check_key(self,key:str):
        key_clean = key.strip().replace(" ","")
        if len(key_clean) == 0:
            print("[!] openai api key not provided")
            sys.exit(-1)


    def __init__(self,dataset_name:str,open_api_key:str = None):
        self.dataset_name = dataset_name
        self._check_key(open_api_key)
        self.open_api_key = open_api_key

        
    
    def _features_preparation(self,df:pd.DataFrame) -> pd.DataFrame:
        feature_df = extract_features_to_csv(df)
        return feature_df
    
    def _dataframe_preparation(self,input_path:str) -> pd.DataFrame:
        print(f"[*] converting files to one single dataframe...")
        df = input_file_transformation(input_path)
        print(f"[*] dataframe prepration complete")
        print(f'[*] total columns present in the {self.dataset_name}',df.shape[0])
        return df
    
    def _dataset_preprocessing(self):
        ### Source file path
        folder_path = os.path.join(PATH_CONFIG["CONTAINER_PATH"],PATH_CONFIG["INF_DATA_PATH"],'inference_repo', self.dataset_name)

        ### 1. If you want convert the inference repo to inference processed 
        source_path = source_file_conversion(folder_path)
        ### 2. If you have the processed file already in the inference processed folder
        source_path = folder_path.replace('/inference_repo/','/inference_repo_processed/')
        return source_path

    def model_prediction(self,save_result = False):
        di_start = time.time()

        MODEL_VERSION = '13052023'
        # processes the files within the given dataset name i.e ( cleaning and standardization)
        processed_input_file_path = self._dataset_preprocessing()
        df = self._dataframe_preparation(processed_input_file_path)
        features_df = self._features_preparation(df)

        l1_model = L1Model()
        l3_model = L3Model(openai_api_key=self.open_api_key)

        print("[*] L1 model prediction started...")
        l1_model_prediction = l1_model.model_prediction(features_df,model_version=MODEL_VERSION,process_type="inference")
        print("[*] L1 model prediction complete...")
        
        print("[*] L3 model prediction started...")
        l1_l3_model_prediction = l3_model.model_prediction(l1_pred=l1_model_prediction,feat_df=features_df,df=df)
        di_end = time.time()

        print(f"[*] datatype identification complete... took {round((di_end-di_start)/60,2)} minute")

        if save_result:
            pass

        return l1_l3_model_prediction
    
    def _check_necessary_requirements(self):
        pass
