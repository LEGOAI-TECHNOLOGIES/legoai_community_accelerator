import os.path
import time
from dataclasses import dataclass
from getpass import getpass
from colorama import Fore
import warnings

from legoai.modules.datatype_identification.l1_model import L1Model
from legoai.modules.datatype_identification.l2_model import L2Model
from legoai.modules.datatype_identification.utils import *
from legoai.modules.datatype_identification.functional import extract_features_to_csv
from legoai.core.configuration import check_path_exists_or_not
from legoai.core.utils import *

from tqdm import tqdm

int_data_l1_path = os.path.join(PATH_CONFIG['INT_DATA_PATH'], 'datatype_l1_identification')  ## Intermediate data path
l1_feature_path = os.path.join(PATH_CONFIG['ANALYTICAL_DATA_PATH'],'datatype_l1_identification')  ## Final feature data path


L2_PREDICTION = 'predicted_datatype_l2'
@dataclass
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

        - l2 model:
            l2 model is mainly responsible for classifying the integer and float datatype into integer_measure or
            integer_dimension and same for float,
            this model also finds the specific format the date & time datatype lies in
            (eg. YYYY:MM:DD , YYYY-MM-DD, YYYY:MM:DD H:m:s , etc...)
            this model uses llm for dimension & measure classification and
            a regex based approach for the date & time classification.
    """
    l1_model: L1Model = None
    l2_model: L2Model = None

    @staticmethod
    def prepare_dataset(input_path:str, output_path:str="l1_training_resource"):
        """
        - Method to prepare training file and ground truth file.

        Parameters
        ----------
        input_path (str): directory path where all the training files are present.
        output_path (str): final directory output path where the training file & a ground truth file will be added
        """
        assert not input_path.strip().__eq__(''), "input path cannot be empty"
        assert not output_path.strip().__eq__(''), "output path cannot be empty"
        input_path = os.path.normpath(input_path)
        output_path = os.path.normpath(output_path)

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"{input_path} does not exist")


        json_data = prepare_di_training_file(input_path)
        gt_df = prepare_di_ground_truth(json_data)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        final_training_file_path = os.path.normpath(
            os.path.join(output_path, f"training_l1_{datetime.now().strftime('%d%m%Y')}.json")
        )
        final_gt_file_path = os.path.normpath(
            os.path.join(output_path, f"ground_truth_l1_{datetime.now().strftime('%d%m%Y')}.xlsx")
        )

        with open(final_training_file_path, "w") as file:
            json.dump(json_data, file)

        print(f"[*] Training file created at: {Fore.RED} {final_training_file_path} {Fore.RESET}")
        add_data_validation_excel_gt(gt_df,final_gt_file_path)
        print(f"[*] Ground truth file created at: {Fore.RED} {final_gt_file_path} {Fore.RESET} ... annotate/label the ground truth before proceeding further")
    
    @staticmethod
    def check_dataset(training_path:str='', gt_path:str='',**kwargs):
        """
        - Check the DI(Datatype Identification) Training and GT file for null,empty values , missing columns ( required for DI )
        Parameters
        ----------
        training_path (str): path to the training file
        gt_path (str): path to the ground truth file
        """
        # for displaying success print statements or not
        _display_message = kwargs.get('display',True)

        assert not training_path.strip().__eq__(''), "input path cannot be empty"
        assert not gt_path.strip().__eq__(''), "output path cannot be empty"
        training_path = os.path.normpath(training_path)
        gt_path = os.path.normpath(gt_path)
        if _display_message:
            print("[*] Checking Training data:")
        training_df = precheck_di_training_file(training_path, REQUIRED_TRAINING_COLUMNS,_display_message)
        if _display_message:
            print("\n[*] Checking Ground truth data:")
        gt_df = precheck_di_training_file(gt_path, REQUIRED_GT_COLUMNS, _display_message, file_type='gt' )

        # final check for any missing master_id from training in ground truth
        if not all(training_df['master_id'].isin(gt_df['master_id'])):
            missing_master_id_in_gt = training_df[~(training_df['master_id'].isin(gt_df['master_id']))]['master_id'].values.tolist()
            warnings.warn(f"{','.join(missing_master_id_in_gt)} from training missing in ground truth")
        else:
            if _display_message:
                print(f"[\u2713] Every training data present in ground truth\n")


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

        output_path = os.path.normpath(os.path.join(output_path, "features"))
        feature_df = extract_features_to_csv(df)
        repo_name = feature_df['dataset_name'].tolist()[0].replace("_processed","")
        features_path = os.path.join(output_path,f"di_features_{repo_name}_{datetime.now().strftime('%d%m%Y')}.csv")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
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
                print(f"[*] Meta information saved at:\n {meta_data_save_path}")

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

        if os.path.isdir(data_path):
            for path, subdirs, files in os.walk(data_path):
                for name in files:
                    feats_content.append(os.path.join(path, name))
        elif os.path.isfile(data_path):
            feats_content.append(data_path)


        features_df = pd.DataFrame()

        for i, filename in enumerate(feats_content):
            feats_csv = pd.read_csv(filename,low_memory = False)
            feats_csv['file_name'] = filename.split(os.sep)[1].replace('.csv', '')


            # for handling any type of files
            _meta_df['filename'] = _meta_df['filename'].apply(lambda x: x.split(".")[0])

            feats_csv['repo_name'] = _meta_df[_meta_df['filename'] == filename.split(os.sep)[-1].replace(
                '_feats.csv','')]['reponame'].tolist()[0]

            # feats_csv['repo_name'] = _meta_df[_meta_df['filename'] ==
            # filename.split(os.sep)[-1].replace('_feats.csv','.json')]['reponame'].tolist()[0]
            #
            features_df = pd.concat([features_df, feats_csv])
            # print(len(feats_content) - i)

        ### Create column id for the mapping features to the respective labels
        features_df['column_id'] = features_df.apply(lambda x: generate_id(str(x['id']), x['repo_name']), axis=1)

        ### Store the features dataframe to the feature location
        full_filepath = os.path.join(container_path, l1_feature_path, "di_l1_consolidated_feats_data.csv")
        features_df.to_csv(full_filepath, index=False)
        print(f"[*] Consolidated features saved at:\n {full_filepath}")

    @staticmethod
    def build_l1_model(gt_path:str,training_size:float,validation_size:float,model_version:str=datetime.now().strftime("%d%m%Y"),**kwargs) -> tuple:
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
            model_version=model_version,
            training_size = training_size,
            validation_size = validation_size
        )


        # train test creation
        X_train, y_train, X_validation, y_validation, X_test, y_test = l1_model.train_test_creation(gt_path)

        ### Train the model using train, validation and test data from the feature dataset
        X_test_pred, class_report_test, conf_df_test, X_validation_pred, class_report_val, conf_df_val = l1_model.build_ML_Model(
            X_train, y_train, X_validation, y_validation, X_test, y_test)
        print(f"\n[!] Important Model Version: {Fore.RED} {model_version} {Fore.RESET} , Training path: {Fore.RED} {PATH_CONFIG['CONTAINER_PATH']} {Fore.RESET} these information's are used during inference\n")
        return class_report_test, conf_df_test, class_report_val, conf_df_val

    def __execute_l1_training_pipeline(self, dataset_path: str, gt_path: str,training_size:float,validation_size:float,model_version:str,**kwargs) -> tuple:
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

        # Feature extraction Part

        # 2nd step is to extract the features from the data and store it as intermediate result
        self.extract_features_store_intermediate(_meta_df)

        # 3rd step is to combine the all the features into one consolidated features_file
        self.combine_intermediate_features(_meta_df)

        # Model Building part
        #4th and final step is to build the model by combining features and the gt
        return self.build_l1_model(gt_path,training_size,validation_size,model_version,**kwargs)

    def __execute_inference_pipeline(self,input_path: str,output_path: str):
        """
            Executes the whole datatype identification pipeline.
            - 1st step to prepare the data (also checks for duplicate columns and performs data standardization)
            - 2nd step is to convert all data in one single dataframe
            - 3rd step is to create all the features for model inference
            - 4th step is to run the l1 model prediction
            - 5th step is to run the l2 model prediction

            Parameters:
            ----------
            input_path(str): the path for the dataset
            output_path(str): destination folder where all results will be stored,i.e features, processed files, and final models.

            Returns:
            --------
            l1_l2_model_prediction (pd.DataFrame): the final l1 and l2 combined result.
            or
            l1_model_prediction (pd.DataFrame): only l1 model prediction if openai key not given
        """

        di_start = time.time()
        # 1st step to prepare the data (also checks for duplicate columns and performs data standardization)
        inference_source_path = self.data_preparation(input_path,output_path)

        # 2nd step is to convert all dataset in one single dataframe
        df = self.dataframe_preparation(inference_source_path)
        print(f'[*] Total columns present in the {os.path.split(input_path)[-1]} : ', df.shape[0])

        # 3rd step is to create all the features for model inference
        features_df = self.features_preparation(df, output_path)
        repo_name = features_df['dataset_name'].tolist()[0].replace("_processed","")
        l2_start_time = time.time()
        # 4th step is to run the l1 model prediction
        _model_prediction = self.l1_model.model_prediction(features_df, process_type="inference")
        l1_finish_time = time.time()

        FINAL_SUBSET_COLUMNS = ['master_id', 'repo_name', 'table_name','column_name','predicted_datatype_l1',
                                'predicted_probability_l1']
        # if api key not given only run l1 model
        if self.l2_model is None:
            _model_prediction[['repo_name','table_name','column_name']] = df['master_id'].str.split(r"\$\$##\$\$",regex = True,expand = True)
            # _model_prediction.drop(columns=['master_id'],inplace=True)
            _model_prediction['master_id'] = df['master_id']

        else:
            # 5th step is to run the l2 model prediction
            _model_prediction = self.l2_model.model_prediction(l1_pred=_model_prediction, feat_df=features_df, df=df)

            _model_prediction.rename(columns={'dataset_name':'repo_name'},inplace = True)
            # add  l2 prediction in for final result subset columns
            FINAL_SUBSET_COLUMNS.append(L2_PREDICTION)


        di_end = time.time()
        print(f"[*] Inference complete ... took {round((di_end - di_start) / 60, 2)} minute ...")
        # print(f"[*] L1 took {round((l1_finish_time - di_start) / 60, 2)} minute ")
        # print(f"[*] L2 took {round((di_end - l2_start_time)/60,2)} minute ")
        output_path = os.path.normpath(os.path.join(output_path, "output"))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        final_output_path = os.path.join(output_path, f"di_final_output_{repo_name}_{datetime.now().strftime('%d%m%Y')}.csv")

        # Subset final columns
        _model_prediction = _model_prediction[FINAL_SUBSET_COLUMNS]
        _model_prediction.to_csv(final_output_path,index=False)
        print(f"[*] Final output saved at {final_output_path}")
        return _model_prediction

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
    def run_l2_model(open_api_key: str, l1_model_prediction: pd.DataFrame, features_df: pd.DataFrame,
                     df: pd.DataFrame,output_path:str) -> pd.DataFrame:
        """
        Runs the L2 model separately ( isolated )
        Parameters
        ----------
        open_api_key (str): api key for llm model
        l1_model_prediction (pd.Dataframe): prediction result from l1 model
        feature_df (pd.Dataframe): dataframe consisting of all the features extracted from extract_features_to_csv
        df (pd.Dataframe): dataframe obtained from the dataset

        Returns
        ------
        l1_l2_model_prediction (pd.Dataframe): result from l1 and l2 results combined
        """
        check_openai_key(open_api_key)
        l1_l2_model_prediction = pd.DataFrame()
        try:
            l1_l2_model_prediction = L2Model(openai_api_key=open_api_key).model_prediction(l1_pred=l1_model_prediction,
                                                                                           feat_df=features_df, df=df)
            output_path = os.path.normpath(os.path.join(output_path,"output"))

            # save l2 model output
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            final_output_path = os.path.join(output_path,f"di_final_output_{datetime.now().strftime('%d%m%Y')}.csv")
            l1_l2_model_prediction.to_csv(final_output_path)
            print(f"[*] Final output saved at {final_output_path}")

        except Exception as ex:
            print(ex)
        finally:
            return l1_l2_model_prediction

    @staticmethod
    def extract_features(input_path: str, output_path: str = None) -> pd.DataFrame:
        """
        extract features from the dataset / repo
            
        Parameters
        ---------
        input_path (str): path of the dataset / directory from where you want to extract the features.

        Returns
        -------
        pd.DataFrame: features extracted dataframe.
        """

        folder_path = input_path

        # folder_path = os.path.join(PATH_CONFIG["CONTAINER_PATH"],PATH_CONFIG["INF_DATA_PATH"],'inference_repo', dataset_name)
        if os.path.exists(folder_path):
            parquet_df = input_file_transformation(folder_path)
            try:
                if output_path is not None and bool(output_path.strip()):
                    parquet_df.to_csv(output_path)
            except Exception:
                new_save_path = f"di_features_{datetime.now().strftime('%Y%M%d')}"
                print(f"[*] {output_path} path not found ... saved at {new_save_path} ...")
                parquet_df.to_csv(new_save_path)

            return extract_features_to_csv(parquet_df=parquet_df)
        else:
            print(f"[!] dataset doesn't exists in the path {folder_path}")

    @classmethod
    def training_pipeline(cls,input_path:str=None,gt_path:str=None,output_path:str=None,training_size:float=0.6,validation_size:float=0.2,model_version:str=datetime.now().strftime("%d%m%Y"),**kwargs):
        """
        Parameters
        ----------
        input_path (str): dataset file path for training
        gt_path (str): ground truth path for the datatype for all the columns in one single file
        model_version (str): model version to save under for trained finalized model i.e. ( encoders , and classifier)
        trainig_size (float): training data distribution 0.6 = 60% default
        validation_size (float): validation data distribution 0.2 = 20% default

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

        # check training & ground truth but supress print message. ( for safe failure)
        DataTypeIdentificationPipeline.check_dataset(
            training_path= input_path ,
            gt_path= gt_path,
            display=False
        )

        # check training , test & validation data distribution
        total_data_ratio = 1
        training_size = float(training_size)
        validation_size = float(validation_size)

        assert training_size < 1 and training_size > 0,"Provide valid training size between 0 and 1"
        assert validation_size < 1 and validation_size > 0 and validation_size < (total_data_ratio - training_size),f"Invalid {validation_size} for {training_size} training size"
        test_size = round(total_data_ratio - (training_size + validation_size),2)
        assert  round(training_size + validation_size + test_size,1) == total_data_ratio, "Provide valid training & validation size"


        return cls().__execute_l1_training_pipeline(input_path, gt_path,training_size , validation_size , model_version,  **kwargs)

    @classmethod
    def prediction_pipeline(cls,input_path:str=None, output_path:str="datatype_identification_inference", L2_predict: bool = False,**kwargs):
        """
            Executes the inference pipelines and saves the result and also returns the final result

            Parameters
            ----------
            input_path (str): the directory or file path to the inference dataset.
            output_path (str): output path to save all the results, i.e. processed files,features, and final l1 and l2 model output.
            L2_predict (bool): flag to denote whether to run L2 model or not

            [!note]
            - for custom model prediction
            training_path (str) optional: path to the training directory ( obtained after training_pipeline())
            model_version (str) optional: model number under with the l1 classifier & encoder is saved also obtained from training_pipeline()

            Returns
            -------
            DatatypeIdentification object with l1 and l2 model loaded.
        """

        # ---- For running custom trained model ---
        training_path = kwargs.get("training_path",None)
        model_version = kwargs.get("model_version",None)

        l1_encoder_path = None
        l1_model_path = None

        if training_path and model_version:
            if not os.path.exists(training_path) and not os.path.isdir(training_path):
                raise FileNotFoundError(f"Training path {training_path} is not valid directory")
            else:
                l1_encoder_path = os.path.join(training_path,"model","model_objects","datatype_l1_identification",f"di_l1_classifier_encoder_{model_version}.pkl")
                l1_model_path = os.path.join(training_path,"model","model_objects","datatype_l1_identification",f"di_l1_classifier_{model_version}.pkl")
                if not os.path.exists(l1_encoder_path) or not os.path.exists(l1_model_path):
                    raise FileNotFoundError(f"Training path is invalid required files not found at {training_path}")
                else:
                    l1_encoder_path = os.path.normpath(l1_encoder_path)
                    l1_model_path = os.path.normpath(l1_model_path)
                    print(f"[*] Using custom trained model, Model Version: {Fore.RED} {model_version} {Fore.RESET}")
        # --------- #

        # l1_encoder_path = kwargs.get("encoder_path",None)
        # l1_model_path = kwargs.get("model_path",None)

        _di_pipeline = None
        
        if L2_predict:
            openai_api_key = getpass("[*] Provide openai api key: ")
            check_openai_key(openai_api_key)
            _di_pipeline = cls(
               l1_model = L1Model.for_inference(
                    encoder_path=l1_encoder_path,
                    model_path=l1_model_path
                ),
                l2_model= L2Model(openai_api_key=openai_api_key)
            )
        else:
            # print("[*] OpenAI api key not provided ... inference will only run for L1 datatype identification ...")
            _di_pipeline = cls(l1_model= L1Model.for_inference(
                encoder_path=l1_encoder_path,
                model_path=l1_model_path
            ))
        input_path = os.path.normpath(input_path)
        output_path = os.path.normpath(output_path)
        return _di_pipeline.__execute_inference_pipeline(input_path=input_path,output_path=output_path)
