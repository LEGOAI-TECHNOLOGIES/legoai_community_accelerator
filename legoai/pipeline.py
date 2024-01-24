
from legoai.modules.datatype_identification.utils import source_file_conversion,input_file_transformation
from legoai.modules.datatype_identification.functional import extract_features_to_csv
from legoai.core.path_configuration import PATH_CONFIG
from legoai.modules.datatype_identification.l1_model import L1Model
from legoai.modules.datatype_identification.l3_model import L3Model
from legoai.modules.datatype_identification import MODEL_CONFIG
from legoai.helper import download_default_dataset , check_openai_key
import pandas as pd
import os
import time

class DataTypeIdentificationPipeline:
    '''
    A pipeline that identifies datatypes for specific columns in a dataset.

        Details:
            the Datatype identification pipeline is divided into 3 parts:
                - features extraction:
                    Here, column level features is extracted from each files (like .csv) , the main features includes 
                   ('dateRatio','wordlen_mean','rangeRatio','floatRatio','zero_flag','intRatio','alphaNumRatio','alphaRatio','frac_unique_sample','flag_numcells')
                
                - l1 model:
                    l1 model uses pretrained xgbclassifier that classifies the columns into one of datatype in 
                    ('alphanumeric','close_ended_text','open_ended_text','float','integer', 'date & time', 'others').

                - l3 model:
                    l3 model is mainly responsible for classifying the integer and float datatype into integer_measure or integer_dimension and same for float,
                    this model also finds the specific format the date & time datatype lies in (eg. YYYY:MM:DD , YYYY-MM-DD, YYYY:MM:DD H:m:s , etc...)
                    this model uses llm for dimension & measure classification and a regex based approach for the date & time classification.     

        Attributes:
            dataset_name (str): folder consiting of files (like .csv) whose column datatypes is to be identified.
            open_api_key (str): open api key for l3 model.
    
        Methods:
            run_pretrained_model(save_to:str=None):
                runs the model on pretrained l1 model (xgbclassifier) and l3 (llm model) and saves to specified path
            
            load_example_dataset(open_api_key:str=None):
                loads example dataset ('ecommerce_data') to use pipeline on the dataset.
            
            extract_features(datset_name:str=None):
                extracts necessary features used by l1 model.
            
            run_l1_model(features_df:pd.DataFrame=None):
                solely runs the l1 models from the extracted features.

            run_l3_model(open_api_key:str,l1_model_prediction:pd.DataFrame,features_df:pd.DataFrame,df:pd.DataFrame):
                solely runs the l1 models , from predictions obtained from l1 model , features obtained from extracting the dataset files and
                the dataframe of the dataset itself.
    '''

    def __init__(self,dataset_name:str = None,openai_api_key:str = None) -> None:
        
        if dataset_name is None or openai_api_key is None:
            raise ValueError("[!] dataset name or api key cannot be empty")
        
        check_openai_key(openai_api_key)
        self.dataset_name = dataset_name
        self.openai_api_key = openai_api_key

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
        folder_path = os.path.join(PATH_CONFIG["CONTAINER_PATH"],PATH_CONFIG["INF_DATA_PATH"],self.dataset_name)

        ### 1. If you want convert the inference repo to inference processed 
        source_path = source_file_conversion(folder_path)
        ### 2. If you have the processed file already in the inference processed folder
        # source_path = folder_path.replace('/inference_repo/','/inference_repo_processed/')
        return source_path

    def run_pretrained_pipeline(self,save_to=None):
        '''
        Runs the datatype identification pipeline on the pretrained l1 model.
            Parameters:
                save_to (str): path to save the final output from the pipeline.
            Returns:
                l1_l3_model_prediction (pd.DataFrame): final pipeline result.
        '''

        di_start = time.time()
        
        # 1st part of di pipeline (data preparation)
        # processes the files within the given dataset name i.e ( cleaning and standardization)
        processed_input_file_path = self.__dataset_preprocessing()
        df = self.__dataframe_preparation(processed_input_file_path)

        # 2nd part of di pipeline (features extraction)
        features_df = self.__features_preparation(df)

        # 3rd part of di pipeline (l1 model prediction)
        l1_model = L1Model()
        l1_model_prediction = l1_model.model_prediction(features_df,model_version=MODEL_CONFIG["L1PARAMS"]["MODEL_VERSION"],process_type="inference")

        #final part of the di pipeline ( l3 model prediction)
        l3_model = L3Model(openai_api_key=self.openai_api_key)
        l1_l3_model_prediction = l3_model.model_prediction(l1_pred=l1_model_prediction,feat_df=features_df,df=df)
        
        di_end = time.time()
        print(f"[*] datatype identification pipeline complete... took {round((di_end-di_start)/60,2)} minute")

        if save_to is not None:
            
            try:
                l1_l3_model_prediction.to_csv(save_to)
            except Exception:
                print("[*] couldnot save to specified path...")
                l1_l3_model_prediction.to_csv("di_output.csv")

            print(f"[*] final result saved at {save_to}")
            

        return l1_l3_model_prediction
    

    @classmethod
    def load_example_dataset(cls,open_api_key:str=None):
        '''
        Downloads example dataset ('ecommerce_dataset') , use it to test the pipeline.
        
            Parameters:
                open_api_key (str): your open api key for l3 model.
            Returns:
                object (DatatypeIdentification): instance of datatype identification pipeline.
        '''
        dataset_name = download_default_dataset()
        return cls(dataset_name = dataset_name,openai_api_key = open_api_key)


    # exposing individual endpoints of the datatype identificaion module

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

    @staticmethod
    def run_l1_model(features_df:pd.DataFrame,) -> pd.DataFrame:
        '''
        Runs the L1 model seperately. 
            
            Parameters:
                features_df (pd.DataFrame): dataframe that consists of features extracted from extract_features()
            Returns:
                pd.DataFrame: l1 model prediction.
        '''

        return L1Model().model_prediction(features_df,model_version=MODEL_CONFIG["L1PARAMS"]["MODEL_VERSION"],process_type="inference")


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
        
    


        
        


