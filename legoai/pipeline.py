
from legoai.modules.datatype_identification.utils import source_file_conversion,input_file_transformation
from legoai.modules.datatype_identification.functional import extract_features_to_csv
from legoai.core.path_configuration import PATH_CONFIG
from legoai.modules.datatype_identification.l1_model import L1Model
from legoai.modules.datatype_identification.l3_model import L3Model
import openai

import pandas as pd
import os , sys
import time
import datetime

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


MODEL_VERSION = '13052023'
DEFAULT_DATASET = "ecommerce_data"

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


class DataTypeIdentificationPipeline:

    def __init__(self,dataset_name:str,open_api_key:str = None):
        check_openai_key(open_api_key)
        self.dataset_name = dataset_name
        self.open_api_key = open_api_key

        
    
    def __features_preparation(self,df:pd.DataFrame) -> pd.DataFrame:
        feature_df = extract_features_to_csv(df)
        return feature_df
    
    def __dataframe_preparation(self,input_path:str) -> pd.DataFrame:
        print(f"[*] converting files to one single dataframe...")
        df = input_file_transformation(input_path)
        print(f"[*] dataframe prepration complete")
        print(f'[*] total columns present in the {self.dataset_name}',df.shape[0])
        return df
    
    def __dataset_preprocessing(self):
        ### Source file path
        folder_path = os.path.join(PATH_CONFIG["CONTAINER_PATH"],PATH_CONFIG["INF_DATA_PATH"],'inference_repo', self.dataset_name)

        ### 1. If you want convert the inference repo to inference processed 
        source_path = source_file_conversion(folder_path)
        ### 2. If you have the processed file already in the inference processed folder
        # source_path = folder_path.replace('/inference_repo/','/inference_repo_processed/')
        return source_path

    def run_pipeline(self,save_result = False):
        di_start = time.time()
        
        # 1st part of di pipeline (data preparation)
        # processes the files within the given dataset name i.e ( cleaning and standardization)
        processed_input_file_path = self.__dataset_preprocessing()
        df = self.__dataframe_preparation(processed_input_file_path)


        # 2nd part of di pipeline (features extraction)
        features_df = self.__features_preparation(df)

        # 3rd part of di pipeline (l1 model prediction)
        l1_model = L1Model()
        l1_model_prediction = l1_model.model_prediction(features_df,model_version=MODEL_VERSION,process_type="inference")

        #final part of the di pipeline ( l3 model prediction)
        l3_model = L3Model(openai_api_key=self.open_api_key)
        l1_l3_model_prediction = l3_model.model_prediction(l1_pred=l1_model_prediction,feat_df=features_df,df=df)
        
        di_end = time.time()
        print(f"[*] datatype identification pipeline complete... took {round((di_end-di_start)/60,2)} minute")

        if save_result:
            l1_l3_model_prediction.to_csv(f"di_predicted_{self.dataset_name}_{datetime.now().strftime('%Y%m%d')}")
            print("[*] final result saved...")

        return l1_l3_model_prediction
    

    @classmethod
    def load_default_model(cls,open_api_key:str):

        return cls(
            dataset_name = DEFAULT_DATASET,
            open_api_key = open_api_key
        )


    # exposing individual endpoints of the datatype identificaion module

    @staticmethod
    def extract_features(dataset_name) -> pd.DataFrame:
        folder_path = os.path.join(PATH_CONFIG["CONTAINER_PATH"],PATH_CONFIG["INF_DATA_PATH"],'inference_repo', dataset_name)
        if os.path.isdir(folder_path):
            parquet_df = input_file_transformation(folder_path)
            return extract_features_to_csv(parquet_df=parquet_df)
        else:
            print("[!] dataset doesnot exists in the path {folder_path}")

    @staticmethod
    def run_l1_model(features_df:pd.DataFrame,) -> pd.DataFrame:

        return L1Model().model_prediction(features_df,model_version=MODEL_VERSION,process_type="inference")


    @staticmethod
    def run_l3_model(open_api_key:str,l1_model_prediction:pd.DataFrame,features_df:pd.DataFrame,df:pd.DataFrame) -> pd.DataFrame:
        '''
            Runs the L3 model seperately ( isolated )
            Parameters:
                open_api_key (str): api key for llm model
                l1_model_prediction (pd.Dataframe): prediction result from l1 model
                feature_df (pd.Dataframe): dataframe consisting of all the features extracted from extract_features_to_csv
                df (pd.Dataframe): dataframe obtained after converting the dataset

            Return:
                l1_l3_model_prediction (pd.Dataframe): result from l1 and l3 prediction combined 
        '''
        check_openai_key(open_api_key)
        l1_l3_model_prediction = pd.DataFrame()
        try:
            l1_l3_model_prediction = L3Model(openai_api_key=open_api_key).model_prediction(l1_pred=l1_model_prediction,feat_df = features_df,df=df)
        except Exception as ex:
            print(ex)
        finally:
            return l1_l3_model_prediction
        
    


        
        


