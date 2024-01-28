import pandas as pd

import os
import time

from legoai.modules.datatype_identification.l1_model import L1Model
from legoai.modules.datatype_identification.l3_model import L3Model
from legoai.modules.datatype_identification.utils import source_file_conversion , input_file_transformation
from legoai.modules.datatype_identification.functional import extract_features_to_csv
from legoai.core.configuration import PATH_CONFIG
from legoai.utils import check_openai_key,download_default_dataset


class DataTypeIdentificationPipeline:
    '''A pipeline that identifies datatypes for specific columns in a dataset.
        
        Details:
        the Datatype identification pipeline is divided into 3 parts:
        - features extraction:
            Here, column level features is extracted from each files (like .csv) , the main features includes 
            ('dateRatio','wordlen_mean','rangeRatio','floatRatio','zero_flag','intRatio','alphaNumRatio','alphaRatio','frac_unique_sample','flag_numcells')
                
        - l1 model:
            l1 model uses pretrained xgbclassifier that classifies the columns into one of datatype in 
            ('alphanumeric','close_ended_text','open_ended_text','float','integer', 'date & time' , 'others').

        - l3 model:
            l3 model is mainly responsible for classifying the integer and float datatype into integer_measure or integer_dimension and same for float,
            this model also finds the specific format the date & time datatype lies in (eg. YYYY:MM:DD , YYYY-MM-DD, YYYY:MM:DD H:m:s , etc...)
            this model uses llm for dimension & measure classification and a regex based approach for the date & time classification.     
    '''
    
    def __init__(self,l1_model:L1Model=None,l3_model:L3Model=None):
        self.l1_model = l1_model
        self.l3_model = l3_model
        

    @classmethod
    def pretrained_pipeline(cls,open_api_key:str=None):
        check_openai_key(open_api_key)

        return cls(
            L1Model.load_pretrained(),
            L3Model(openai_api_key=open_api_key)
        )

    def __dataset_preprocessing(self,dataset_name:str):
        folder_path = os.path.join(PATH_CONFIG["CONTAINER_PATH"],PATH_CONFIG["INF_DATA_PATH"],dataset_name)
        source_path = source_file_conversion(folder_path)
        return source_path
    
    def __dataframe_preparation(self,input_path:str) -> pd.DataFrame:
        df = input_file_transformation(input_path)
        return df

    def __features_preparation(self,df:pd.DataFrame) -> pd.DataFrame:
        feature_df = extract_features_to_csv(df)
        return feature_df
    
    def __execute_pipeline(self,dataset_name:str):
        di_start = time.time()
        # 1st step to preprocess the dataset for duplicate columns and data standardization
        inference_source_path = self.__dataset_preprocessing(dataset_name)

        # 2nd step is to convert all dataset in one single dataframe
        df = self.__dataframe_preparation(inference_source_path)
        print(f'[*] total columns present in the {dataset_name}',df.shape[0])

        # 3rd step is to create all the features for model inferencing
        features_df = self.__features_preparation(df)

        # 4th step is to run the l1 model prediction
        l1_model_prediction = self.l1_model.model_prediction(features_df,process_type="inference")
        
        #5th step is to run the l3 model preidiction
        l1_l3_model_prediction = self.l3_model.model_prediction(l1_pred=l1_model_prediction,feat_df=features_df,df=df)

        di_end = time.time()
        print(f"[*] datatype identification pipeline complete... took {round((di_end-di_start)/60,2)} minute")

        return l1_l3_model_prediction

    @classmethod
    def load_example_dataset(cls):
        '''
        Downloads example dataset ('ecommerce_dataset') , use it to test the pipeline.
        
            Parameters:
                None
            Returns:
                dataset_name (str): an example dataset 
        '''
        dataset_name = download_default_dataset()
        return dataset_name
    

    @staticmethod
    def run_l1_model(features_df:pd.DataFrame,) -> pd.DataFrame:
        '''
        Runs the L1 model seperately. 
            
            Parameters:
                features_df (pd.DataFrame): dataframe that consists of features extracted from extract_features()
            Returns:
                pd.DataFrame: l1 model prediction.
        '''

        return L1Model.load_pretrained().model_prediction(features_df,process_type="inference")
    
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
        
    @staticmethod
    def extract_features(dataset_name:str) -> pd.DataFrame:
        '''
        extract features from the dataset / repo
            
            Parameters:
                dataset_name (str): name of the dataset / directory from where you want to extract the features.
            Returns:
                pd.DataFrame: features extracted dataframe.
        '''

        folder_path = os.path.join(PATH_CONFIG["CONTAINER_PATH"],PATH_CONFIG["INF_DATA_PATH"],'inference_repo', dataset_name)
        if os.path.isdir(folder_path):
            parquet_df = input_file_transformation(folder_path)
            return extract_features_to_csv(parquet_df=parquet_df)
        else:
            print("[!] dataset doesnot exists in the path {folder_path}")
            

    def __call__(self,dataset_name:str=None,save_to:str=None):
        '''
            executes the pipelines and saves the result and also returns the final result

            Parameters:
                dataset_name (str): the dataset name that is defined under the path in .env file
                save_to (str): destination path to save the final result.
            Returns
                result (pd.DataFrame): final result dataframe
        '''
        dataset_path =  os.path.join(PATH_CONFIG["CONTAINER_PATH"],PATH_CONFIG["INF_DATA_PATH"],dataset_name)
        
        if os.path.isdir(dataset_path):
            result = self.__execute_pipeline(dataset_name)    
            if save_to is not None:
            
                try:
                    result.to_csv(save_to)
                except Exception:
                    print("[*] couldnot save to specified path...")
                    result.to_csv("di_output.csv")

                print(f"[*] final result saved at {save_to}")
            return result
        else:
            raise FileNotFoundError(f"[!] no such directory as {dataset_path}")

