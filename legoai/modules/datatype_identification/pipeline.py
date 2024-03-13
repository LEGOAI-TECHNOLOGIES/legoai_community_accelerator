import pandas as pd

import traceback
import os
import time
from datetime import datetime
from pathlib import Path

from legoai.modules.datatype_identification.l1_model import L1Model
from legoai.modules.datatype_identification.l3_model import L3Model
from legoai.modules.datatype_identification.utils import *
from legoai.modules.datatype_identification.functional import extract_features_to_csv
from legoai.core.configuration import check_path_exists_or_not, MODEL_CONFIG, PATH_CONFIG
from legoai.utils import *

from tqdm import tqdm

int_data_l1_path = os.path.join(PATH_CONFIG['INT_DATA_PATH'], 'datatype_l1_identification')  ## Intermediate data path
l1_feature_path = os.path.join(PATH_CONFIG['ANALYTICAL_DATA_PATH'],'datatype_l1_identification')  ## Final feature data path
class DataTypeIdentificationPipeline:
    """
        A pipeline that identifies datatype for specific columns in a dataset.
        
        Details:
        the Datatype identification pipeline is divided into 3 parts:
        - features extraction:
            Here, column level features is extracted from each file (like .csv) , the main features includes
            ('dateRatio','wordlen_mean','rangeRatio','floatRatio','zero_flag','intRatio','alphaNumRatio','alphaRatio','frac_unique_sample','flag_numcells')
                
        - l1 model:
            l1 model uses pretrained XGBClassifier that classifies the columns into one of datatype in
            ('alphanumeric','close_ended_text','open_ended_text','float','integer', 'date & time' , 'range_type','others').

        - l3 model:
            l3 model is mainly responsible for classifying the integer and float datatype into integer_measure or
            integer_dimension and same for float,
            this model also finds the specific format the date & time datatype lies in
            (eg. YYYY:MM:DD , YYYY-MM-DD, YYYY:MM:DD H:m:s , etc...)
            this model uses llm for dimension & measure classification and
            a regex based approach for the date & time classification.
    """

    def __init__(self, l1_model: L1Model = None, l3_model: L3Model = None):
        self.l1_model = l1_model
        self.l3_model = l3_model

    @classmethod
    def pretrained_pipeline(cls, openai_api_key: str = None,**kwargs):
        """
            - Returns an object with preloaded L1 model and pre instantiated L3 model
            - if openai api key not given only instantiates with L1 model
            - encoder and model path can be given to run custom model, else default settings will be used
            Parameters
            ----------
            openai_api_key (str): an openai api key for L3 model
            encoder_path (str) optional: full path to the encoder (i.e. pickled object).
            model_path (str) optional; full path to the classifier model (i.e. pickled object).

            Returns
            -------
            DatatypeIdentification object with l1 and l3 model loaded.
        """

        l1_encoder_path = kwargs.get("encoder_path",None)
        l1_model_path = kwargs.get("model_path",None)

        if openai_api_key is not None:
            check_openai_key(openai_api_key)
            return cls(
                L1Model.for_inference(
                    encoder_path=l1_encoder_path,
                    model_path=l1_model_path
                ),
                L3Model(openai_api_key=openai_api_key)
            )

        print("[*] OpenAI api key not provided ... inference will only run for l1 datatype identification ...")
        return cls(L1Model.for_inference(
            encoder_path=l1_encoder_path,
            model_path=l1_model_path
        ), None)


    @staticmethod
    def data_preparation(dataset_path: str,output_path:str):
        """
        - Data preparation for inference

        Parameters
        ----------
        dataset_path (str): path to the dataset folder

        Returns
        -------
        Returns path of new preprocessed data folder
        """
        check_dataset_path(dataset_path)
        output_path = os.path.normpath(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        source_path = source_file_conversion(dataset_path,output_path)
        print(f"[*] Processed files saved at {source_path}")

        return source_path

    @staticmethod
    def dataframe_preparation(input_path: str) -> pd.DataFrame:
        """
        - Converts preprocessed data into one single dataframe

        Parameters
        ----------
        input_path (str): path to the preprocessed data folder ( can be obtained from data_preparation())

        Returns
        -------
        dataframe with all tables combined with columns , values and other relevant information.
        """

        df = input_file_transformation(input_path)
        return df

    @staticmethod
    def features_preparation(df: pd.DataFrame,output_path:str) -> pd.DataFrame:
        """
        - Extract the features from the combined dataframe obtained through dataframe_preparation()

        Parameters
        ----------
        df (pd.DataFrame): Dataframe with all relevant information such as master_id, table_name ,column_name and column_values.
        output_path (str): destination path to store created features.

        Returns
        -------
        Feature extracted dataframe
        """
        output_path = os.path.normpath(os.path.join(output_path,"features"))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        feature_df = extract_features_to_csv(df)
        features_path = os.path.join(output_path,f"intermediate_features_{datetime.now().strftime('%d%m%Y')}.csv")
        feature_df.to_csv(features_path)
        print(f"[*] Features saved at {features_path}")
        return feature_df

    @staticmethod
    def extract_features_store_intermediate(_meta_df: pd.DataFrame,):
        """
        Extract features from relevant columns along with their meta information included and store it in intermediate folder
        - Iterate through each filename and filepath from the raw data, to read the file
        - File read from the raw folder, converted into dataframe format
        - Checks the dataframe for the metadata presence, if not creates the metadata
        - Finally the dataframe is passed as input for feature creation
        - Returned feature df is stored as csv file in the intermediate folder path

        Parameters
        ----------
        _meta_df (pd.DataFrame): dataframe with dataset meta information

        Returns
        -------
        """
        output_path = PATH_CONFIG.get('CONTAINER_PATH')  ## Container Path

        for row in tqdm(range(len(_meta_df))):

            try:
                ## Getting the required variable from the data
                reponame, filename, filepath = _meta_df[['reponame', 'filename', 'filepath']].iloc[row].tolist()

                ### Extracting data from json into dataframe format
                json_df = data_conversion(filename, filepath)

                ### Check if the file is empty or not
                if json_df.shape[0] == 0:
                    continue

                ### Extracting meta data information from the json file
                meta_json_df = meta_information_check(json_df, filename, reponame)
                print(f'\n[*] Meta Data Row Count {filename}: ', meta_json_df.shape)

                ### Extracting features from dataframe
                df = extract_features_to_csv(meta_json_df)

                ### Saving the output to the folder
                full_filepath = filename.replace('.json', '.csv').replace('.parquet', '.csv').replace('.xlsx',
                                                                                                      '.csv').replace(
                    '.txt', '.csv').replace('.csv', '_feats.csv')

                meta_data_save_path = os.path.join(output_path, int_data_l1_path, full_filepath)

                df.to_csv(meta_data_save_path,index =False)
                print(f"[*] Meta information saved at {meta_data_save_path}...")

            except Exception as e:
                print(traceback.format_exc())
                print(e)

    @staticmethod
    def combine_intermediate_features(_meta_df: pd.DataFrame):
        """
        Parameters
        ----------
        _meta_df (pd.DataFrame): dataframe consisting dataset meta information
        # data_path (str): intermediate feature stored folder path ( after extracting features from all files)
        Returns
        -------
        None
        """
        container_path = PATH_CONFIG['CONTAINER_PATH']

        print("[*] Combining all features into single file ...")
        feats_content = []
        data_path = os.path.join(container_path,int_data_l1_path)

        for path, subdirs, files in os.walk(data_path):
            for name in files:
                feats_content.append(os.path.join(path, name))
        features_df = pd.DataFrame()

        for i, filename in enumerate(feats_content):
            feats_csv = pd.read_csv(filename)
            feats_csv['file_name'] = filename.split(os.sep)[1].replace('.csv', '')
            feats_csv['repo_name'] = _meta_df[_meta_df['filename'] ==
            filename.split(os.sep)[-1].replace('_feats.csv', '.json')]['reponame'].tolist()[0]
            features_df = pd.concat([features_df, feats_csv])
            # print(len(feats_content) - i)

        ### Create column id for the mapping features to the respective labels
        features_df['column_id'] = features_df.apply(lambda x: generate_id(str(x['id']), x['repo_name']), axis=1)

        ### Store the features dataframe to the feature location
        full_filepath = os.path.join(container_path, l1_feature_path, "di_l1_consolidated_feats_data.csv")
        features_df.to_csv(full_filepath, index=False)
        print(f"[*] Consolidated features saved at {full_filepath}")

    @staticmethod
    def build_l1_model(gt_path:str,model_version:str=datetime.now().strftime("%d%m%Y"),**kwargs) -> tuple:
        """
        Parameters
        ----------
        gt_path (str): ground truth or label path for the training data
        model_version (str): final model version that is to be stored (for classifier and encoder) if not given today's date will be used.

        Returns
        -------
        classification report and prediction result of  such as confusion matrix test and validation dataset
        """

        l1_model = L1Model.for_training(
            model_version=model_version
        )


        # train test creation
        X_train, y_train, X_validation, y_validation, X_test, y_test = l1_model.train_test_creation(gt_path)

        ### Train the model using train, validation and test data from the feature dataset
        X_test_pred, class_report_test, conf_df_test, X_validation_pred, class_report_val, conf_df_val = l1_model.build_ML_Model(
            X_train, y_train, X_validation, y_validation, X_test, y_test)
        return X_test_pred, class_report_test, conf_df_test, X_validation_pred, class_report_val, conf_df_val

    def __execute_l1_training_pipeline(self, dataset_path: str, gt_path: str,model_version:str,**kwargs) -> tuple:
        """
        runs all the l1 training process at once

        Parameters
        ----------
        dataset_path (str): raw dataset path for training
        gt_path (str): gt file for the datatypes for all the columns in one single file

        Returns
        -------
        tuples: class report and prediction result of test and validation dataset
        """

        # Data processing Part

        # 1st step extract file meta information
        _meta_df = extract_file_meta_info(dataset_path)
        _meta_df.to_csv("D:\metadata.csv")
        # Feature extraction Part

        # 2nd step is to extract the features from the data and store it as intermediate result
        self.extract_features_store_intermediate(_meta_df)

        # 3rd step is to combine the all the features into one consolidated features_file
        self.combine_intermediate_features(_meta_df)

        # Model Building part
        #4th and final step is to build the model by combining features and the gt
        return self.build_l1_model(gt_path,model_version,**kwargs)

    def __execute_inference_pipeline(self,input_path: str,output_path: str):
        """
            Executes the whole datatype identification pipeline.
            - 1st step to prepare the data (also checks for duplicate columns and performs data standardization)
            - 2nd step is to convert all data in one single dataframe
            - 3rd step is to create all the features for model inference
            - 4th step is to run the l1 model prediction
            - 5th step is to run the l3 model prediction

            Parameters:
            ----------
            input_path(str): the path for the dataset
            output_path(str): destination folder where all results will be stored,i.e features, processed files, and final models.

            Returns:
            --------
            l1_l3_model_prediction (pd.DataFrame): the final l1 and l3 combined result.
            or
            l1_model_prediction (pd.DataFrame): only l1 model prediction if openai key not given
        """

        di_start = time.time()
        # 1st step to prepare the data (also checks for duplicate columns and performs data standardization)
        inference_source_path = self.data_preparation(input_path,output_path)

        # 2nd step is to convert all dataset in one single dataframe
        df = self.dataframe_preparation(inference_source_path)
        print(f'[*] Total columns present in the {os.path.split(input_path)[-1]}', df.shape[0])

        # 3rd step is to create all the features for model inference
        features_df = self.features_preparation(df, output_path)

        # 4th step is to run the l1 model prediction
        _model_prediction = self.l1_model.model_prediction(features_df, process_type="inference")

        # if api key not given only run l1 model
        if self.l3_model is None:
            _model_prediction[['repo','table','column']] = df['master_id'].str.split(r"\$\$##\$\$",regex = True,expand = True)
            _model_prediction.drop(columns=['master_id'],inplace=True)

        else:
            # 5th step is to run the l3 model prediction
            _model_prediction = self.l3_model.model_prediction(l1_pred= _model_prediction, feat_df=features_df, df=df)

        di_end = time.time()
        print(f"[*] Inference complete ... took {round((di_end - di_start) / 60, 2)} minute ...")
        output_path = os.path.normpath(os.path.join(output_path, "output"))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        final_output_path = os.path.join(output_path, f"di_final_output_{datetime.now().strftime('%d%m%Y')}.csv")
        _model_prediction.to_csv(final_output_path)
        print(f"[*] Final output saved at {final_output_path}")
        return _model_prediction

    @staticmethod
    def load_example_dataset():
        """
        Downloads example dataset ('ecommerce_dataset') , can be used to test the pipeline.
        Parameters:
        ----------

        Returns:
        -------
        dataset_path (str): an example dataset full path
        """
        dataset_path = download_default_dataset()
        return dataset_path

    @staticmethod
    def run_l1_model(features_df: pd.DataFrame,output_path:str) -> pd.DataFrame:
        """
        Runs the L1 model.
        Parameters
        ----------
        features_df (pd.DataFrame): dataframe that consists of features extracted from extract_features()
        model_version (str): l1 model version that was trained

        Returns
        -------
        pd.DataFrame: l1 model prediction.
        """

        l1_model = L1Model.for_inference(

        ).model_prediction(features_df, process_type="inference")

        output_path = os.path.normpath(os.path.join(output_path,"output"))

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        final_output_path = os.path.join(output_path, "output", f"di_output_l1_model_{datetime.now().strftime('%d%m%Y')}.csv")
        l1_model.to_csv(final_output_path)
        print(f"[*] L1 prediction output saved at {final_output_path}")

    @staticmethod
    def run_l3_model(open_api_key: str, l1_model_prediction: pd.DataFrame, features_df: pd.DataFrame,
                     df: pd.DataFrame,output_path:str) -> pd.DataFrame:
        """
        Runs the L3 model separately ( isolated )
        Parameters
        ----------
        open_api_key (str): api key for llm model
        l1_model_prediction (pd.Dataframe): prediction result from l1 model
        feature_df (pd.Dataframe): dataframe consisting of all the features extracted from extract_features_to_csv
        df (pd.Dataframe): dataframe obtained from the dataset

        Returns
        ------
        l1_l3_model_prediction (pd.Dataframe): result from l1 and l3 results combined
        """
        check_openai_key(open_api_key)
        l1_l3_model_prediction = pd.DataFrame()
        try:
            l1_l3_model_prediction = L3Model(openai_api_key=open_api_key).model_prediction(l1_pred=l1_model_prediction,
                                                                                           feat_df=features_df, df=df)
            output_path = os.path.normpath(os.path.join(output_path,"output"))

            # save l3 model output
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            final_output_path = os.path.join(output_path,f"di_final_output_{datetime.now().strftime('%d%m%Y')}.csv")
            l1_l3_model_prediction.to_csv(final_output_path)
            print(f"[*] Final output saved at {final_output_path}")

        except Exception as ex:
            print(ex)
        finally:
            return l1_l3_model_prediction

    @staticmethod
    def extract_features(dataset_path: str, save_to: str = None) -> pd.DataFrame:
        """
        extract features from the dataset / repo
            
        Parameters
        ---------
        dataset_path (str): path of the dataset / directory from where you want to extract the features.

        Returns
        -------
        pd.DataFrame: features extracted dataframe.
        """

        folder_path = dataset_path

        # folder_path = os.path.join(PATH_CONFIG["CONTAINER_PATH"],PATH_CONFIG["INF_DATA_PATH"],'inference_repo', dataset_name)
        if os.path.exists(folder_path):
            parquet_df = input_file_transformation(folder_path)
            try:
                if save_to is not None:
                    parquet_df.to_csv(save_to)
            except Exception:
                new_save_path = f"di_features_{datetime.now().strftime('%Y%M%d')}"
                print(f"[*] {save_to} path not found ... saved as {new_save_path} ...")
                parquet_df.to_csv(new_save_path)

            return extract_features_to_csv(parquet_df=parquet_df)
        else:
            print(f"[!] dataset doesn't exists in the path {folder_path}")

    def train(self,input_path:str=None,gt_path:str=None,output_path:str=None,model_version:str=datetime.now().strftime("%d%m%Y"),**kwargs):
        """
        Parameters
        ----------
        input_path (str): raw dataset path for training
        gt_path (str): ground truth path for the datatype for all the columns in one single file
        model_version (str): model version to save under for trained finalized model i.e. ( encoders , and classifier)


        Returns
        -------
        tuples: class report and prediction result of test and validation dataset
        """


        # now use path from .env as container path if output path not given
        if output_path is not None and bool(output_path.strip()):
            output_path = os.path.normpath(output_path)
            if bool(output_path.strip()):
                PATH_CONFIG["CONTAINER_PATH"] = os.path.normpath(output_path)

        check_path_exists_or_not(PATH_CONFIG)


        if not bool(input_path and bool(input_path.strip())) or not bool(gt_path and bool(gt_path.strip())):
            raise Exception("[!] Provide valid input path or the ground truth path...")

        input_path = os.path.normpath(input_path)
        gt_path = os.path.normpath(gt_path)
        check_dataset_path(input_path, gt_path)

        return self.__execute_l1_training_pipeline(input_path, gt_path, model_version, **kwargs)

    def predict(self, input_path: str = None, output_path: str = "datatype_identification"):
        """
        Executes the inference pipelines and saves the result and also returns the final result
        Parameters
        ----------
        input_path (str): the path to the inference dataset.
        output_path (str): output path to save all the results, i.e. processed files,features, and final l1 and l3 model output.

        Returns
        -------
        result (pd.DataFrame): final result dataframe
        """
        result = self.__execute_inference_pipeline(input_path,output_path)
        return result
