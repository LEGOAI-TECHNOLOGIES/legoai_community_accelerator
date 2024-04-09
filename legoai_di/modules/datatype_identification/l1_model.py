# ====================================================================
#  Importing the required python packages
# ====================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold

import os

from legoai_di.core.configuration import MODEL_CONFIG,PATH_CONFIG
from legoai_di.core.utils import combine_gt_file


# Creating an object
# #logger = #logger.get#logger(
#     parent_folder_name= 'datatype_l1_identification' , child_folder_name='model'
# )
 

# container_path = PATH_CONFIG["CONTAINER_PATH"]
DEFAULT_MODEL_VERSION = "13052023"

# check for valid encoders and models
ENCODERS = (LabelEncoder,)
MODELS = (XGBClassifier,RandomForestClassifier,SVC,VotingClassifier)


class L1Model:
    """
    L1 model uses pretrained XGBClassifier that classifies the columns into one of datatype in
    ('alphanumeric','close_ended_text','open_ended_text','float','integer', 'date & time' , 'others').
    """
    
    ### Setting the path for the various path for model related info
    def __init__(self,process_type:str = "inference",**kwargs):
        # self.container_path = PATH_CONFIG["CONTAINER_PATH"]

        if process_type.__eq__("inference"):
            self.clf_model = joblib.load(kwargs.get("model_path", None))
            self.encoder = joblib.load(kwargs.get("encoder_path", None))
            # check is encoders and model's are valid object
            assert isinstance(self.clf_model, MODELS), f"[!] provide valid model path ... can be one of {MODELS}..."
            assert isinstance(self.encoder, ENCODERS), f"[!] provide valid encoder path ... can be one of {ENCODERS}..."

        elif process_type.__eq__("training"):
            # PATH_CONFIG = kwargs.get("config",{})
            self.container_path = PATH_CONFIG["CONTAINER_PATH"]
            self.model_version = str(kwargs.get("model_version"))

            self.model_objects_directory = os.path.join(PATH_CONFIG['MODEL_OBJECTS_PATH'], 'datatype_l1_identification')
            self.model_results_directory = os.path.join(PATH_CONFIG['MODEL_RESULTS_PATH'], 'datatype_l1_identification')
            self.model_metrics_directory = os.path.join(PATH_CONFIG['MODEL_METRICS_PATH'], 'datatype_l1_identification')
            self.data_directory = os.path.join(PATH_CONFIG['ANALYTICAL_DATA_PATH'], 'datatype_l1_identification')

            print(f"[*] MODEL VERSION: {self.model_version}")

    @classmethod
    def for_inference(cls, **kwargs):
        """
        - Checks if model version is provided or not as argument
        - If not checks if model version is provided in config file
        - Else choose DEFAULT MODEL VERSION

        Parameters
        ----------
        model_version (str): version of pre-trained classifier model to use

        Returns
        -------
        instance of L1Model with model version
        """

        encoder_path = kwargs.get("encoder_path",None)
        model_path = kwargs.get("model_path",None)

        # if encoder and model path not given load the default model and encoder
        if encoder_path is None or model_path is None:
            dir_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
            default_model_path = os.path.join(os.path.split(dir_path)[0], "model", "model_objects",
                                              'datatype_l1_identification')

            model_path = os.path.join(default_model_path, 'di_l1_classifier_xgb_' + DEFAULT_MODEL_VERSION + '.pkl')
            encoder_path = os.path.join(default_model_path,
                                        'di_l1_classifier_encoder_' + DEFAULT_MODEL_VERSION + '.pkl')

        elif not os.path.isfile(encoder_path) or not os.path.isfile(model_path):
            raise FileNotFoundError("[!] Provide a valid encoder or model path ...")

            # model_version = MODEL_CONFIG.get("L1PARAMS",DEFAULT_MODEL_VERSION).get("MODEL_VERSION",DEFAULT_MODEL_VERSION)

        return cls(process_type="inference", encoder_path=encoder_path, model_path=model_path)

    @classmethod
    def for_training(cls,model_version:str=datetime.now().strftime('%d%m%Y')):
        return cls(process_type="training",model_version=model_version)


    @staticmethod
    def _group_labels(datatype: str) -> str:
        """
        - Groups the similar datatypes into one single type

        Parameters
        ----------
        datatype (str): input datatype

        Returns
        -------
        str: grouped datatype
        """

        # groups the similar datatypes into one single type
        if datatype in ['Short_Integer', 'Long_Integer']:
            return 'Integer'
        elif datatype in ['Short_Float', 'Long_Float']:
            return 'Float'
        elif datatype in ['Short_Alphanumeric', 'Long_Alphanumeric']:
            return 'Alphanumeric'
        elif datatype in ['Open_ended_long_text', 'Open_ended_short_text']:
            return 'Open_ended_text'
        elif datatype == 'Close_ended_short_text':
            return 'Close_ended_text'
        else:
            return datatype

    @staticmethod
    def _feature_subset() -> list:
        """
        - Returns the list of required feature for the modeling

        Returns
        -------
        list: list of required features for modeling
        """
        ##logger.debug('Subsetting the char, par, word, rest features')
        feats_cols = ['dateRatio', 'wordlen_mean', 'rangeRatio', 'floatRatio', 'zero_flag', 'intRatio', 'alphaNumRatio', 'alphaRatio', 'frac_unique_sample','flag_numcells']
        
        # Returns the list of required features for the modeling
        return feats_cols

    def train_test_creation(self, label_path:str,label: str = 'grouped_datatype') -> tuple: #model_version = datetime.now().strftime('%d%m%Y')):
        """
        - Reading the required features file and label data from feature and GT path
        - Merge the feature and label data to create grouped datatype
        - Split the data into train, test and validation for model training
        - Encoding the grouped datatype for modeling and store the encoder object
        - Return train, test and validation dataset for modeling

        Parameters
        ----------
        label_path (str): ground truth for datatype
        label (str): datatype

        Returns
        -------
        tuple: train , test and validation data
        """
        # Reading the required features file from feature path
        #logger.debug('Reading features data')
        feats_path = os.path.join(self.container_path, self.data_directory,'di_l1_consolidated_feats_data.csv')
        
        # Reading the required label file from GT path
        #logger.debug('Reading label data')
        
        # label_path = os.path.join(container_path, self.label_directory,'di_l1_ground_truth.csv')


        # Reading the required feature and label file from path
        model_data = pd.read_csv(feats_path,low_memory=False)



        # try:
            # for default cases
            # label_path = os.path.join(container_path, label_path, 'di_l1_ground_truth.csv')
            # labelled_data = pd.read_csv(label_path)
        # except Exception:
            # for if user provides full path for gt
            #labelled_data = pd.read_csv(label_path)

        # label_path = os.path.join(container_path, label_path)
        labelled_data = combine_gt_file(label_path)

        # convert master id to lower to remove any join inconsistency
        model_data['master_id'] = model_data['master_id'].str.lower()
        labelled_data['master_id'] = labelled_data['master_id'].str.lower()

        print(f'[*] Features: {model_data.shape} , Labels: {labelled_data.shape}')
        # Merge the feature and label data to create grouped datatype
        model_label_data = pd.merge(model_data,labelled_data[['master_id','datatype']],on='master_id')

        assert model_label_data.shape[0] != 0, f"[!] Final Labelled data shape is zero ... check master_id ( ground truth file )"

        print(f'[*] Final Merged Features and Labels: {model_label_data.shape}')
        #logger.debug('Grouping datatype')        
        model_label_data['grouped_datatype'] = model_label_data['datatype'].apply(self._group_labels)
        model_label_data['grouped_datatype'] = model_label_data['grouped_datatype'].str.lower()
        
        #logger.debug('Train - Test dataset creation started...')

        ### Train test and validation split
        train_data, valid_data, test_data = np.split(model_label_data.sample(frac=1), [int(.6 * len(model_label_data)), int(.8 * len(model_label_data))])

        ### Train/Test Data split
        # train_file_name = [MODEL_CONFIG['L1_PARAMS']['TRAIN_DATASET'],MODEL_CONFIG['L1_PARAMS']['VALIDATION_DATASET']]
        #
        # print(model_label_data['repo_name'].unique())
        #
        # repo_name = 'repo_name' if 'repo_name' in model_label_data.columns else 'dataset_name'
        #
        #
        # # Split the data into train, test and validation for model training
        # train_val_data = model_label_data[model_label_data[repo_name].isin(train_file_name)]
        # print(train_val_data['repo_name'].unique())
        # train_data = train_val_data[train_val_data[repo_name] == MODEL_CONFIG['L1_PARAMS']['TRAIN_DATASET']]
        # valid_data = train_val_data[train_val_data[repo_name] == MODEL_CONFIG['L1_PARAMS']['VALIDATION_DATASET']]
        # test_data = model_label_data[~model_label_data[repo_name].isin(train_file_name)]
        #
        print('[*] Train: ',train_data.shape, 'Valid: ',valid_data.shape, 'Test: ',test_data.shape)
        
        ### Train - Test - Validation Set
        X_train = train_data
        y_train = train_data[label].values.flatten()
        
        X_validation = valid_data
        y_validation = valid_data[label].values.flatten()
        
        X_test = test_data
        y_test = test_data[label].values.flatten()
        
        ### Encoding the target variables
        num_classes = len(set(y_train))
        ### check if test , train or validation data are empty
        # assert X_train.shape[0] != 0 or y_train.shape[0] != 0, f"[!] Train data is empty ... is {MODEL_CONFIG['L1_PARAMS']['TRAIN_DATASET']} proper dataset ?"
        # assert X_validation.shape[0] != 0 or y_validation.shape[0] != 0, f"[!] Validation data is empty ... is {MODEL_CONFIG['L1_PARAMS']['VALIDATION_DATASET']} proper dataset ?"
        # assert X_test.shape[0] != 0 or y_test.shape[0] !=0, "[!] Test data is empty ... check train and validation dataset in config.yaml file or properly pass it as training argument"

        encoder = LabelEncoder()
        # encoder.fit(y_train)
        encoder.fit(model_label_data[label].values.flatten())
        # Encoding the label data for the modeling 
        y_train = encoder.transform(y_train)
        y_validation = encoder.transform(y_validation)
        y_test = encoder.transform(y_test)
        
        ### Storing the label encoder object
        model_path = os.path.join(self.container_path, self.model_objects_directory,'di_l1_classifier_encoder_'+self.model_version+'.pkl')
        with open(model_path,'wb') as enc_file:
            joblib.dump(encoder, enc_file)
            print(f"[*] Encoder saved at {model_path} ...")

        #logger.debug('Train - Test dataset creation completed...')

        ### Return train, test and validation dataset for modeling
        return X_train, y_train, X_validation, y_validation, X_test, y_test
    
    # ====================================================================
    # optimize_model: 

    # Parameters: 
    #    X_train - Feature data
    #    y_train - Target data
    # ====================================================================             
    @staticmethod
    def optimize_model(X_train: pd.DataFrame, y_train: np.array, k: int = 3, params: dict = None) -> dict:
        """
        - Define the possible parameters for the model optimization
        - Perform the Grid search for the best model parameter
        - Return the best parameters from optimization

        Parameters
        ----------
        X_train (pd.DataFrame): Feature data
        y_train (pd.DataFrame): Target data
        k (int): no of folds for KFold cross validation
        params (dict): dictionary of parameters for optimization ( same as used in GridSearch)

        Returns
        -------
        best parameters obtained through grid search
        """
        
        # Parameters for the model optimization
        if params is None or params == {}:
            params = {
                    'n_estimators': [100, 200, 500],
                    'criterion': ['gini', 'entropy'],
                    'min_samples_split': [1,2,4,5],
                    'min_samples_leaf': [1,2,4,5],
                    'max_leaf_nodes': [4,10,20,50,None]
                  }
        
        # Grid search for the best model parameter
        gridSearchCV = GridSearchCV(XGBClassifier(n_jobs=-1), params, n_jobs=-1, cv=KFold(n_splits=k))
        gridSearchCV.fit(X_train, y_train)
    
        #logger.debug('Best score:', gridSearchCV.best_score_)
        #logger.debug('Best score:', gridSearchCV.best_params_)    
        
        return gridSearchCV.best_params_

    @staticmethod
    def model_metrics(self, actual, pred, type = 'test'): #model_version = datetime.now().strftime('%d%m%Y')):
        """
        - Calculate the model metrics such as classification report and confusion matrix
        - Perform the Grid search for the best model parameter
        - Return the best parameters from optimization

        Parameters
        ----------
        actual (pd.DataFrame): Actual value of the target
        pred (pd.DataFrame): Predicted value of the target
        types (str): input model name

        Returns
        -------

        """
        #logging.debug('Metrics calculation started')
        types = f"di_l1_classifier_{type}_predicted_xgb"

        # Calculate the classification report for the input actual and predicted
        report = classification_report(actual, pred, output_dict=True)        
        class_report = pd.DataFrame(report).transpose() 
        
        # Calculate the confusion matrix for the input actual and predicted
        conf_mat = confusion_matrix(actual, pred)
        conf_df = pd.DataFrame(conf_mat)
        
        # Path for storing the classification report and confusion matrix report for the input actual and predicted
        report_path = os.path.join(self.container_path, self.model_metrics_directory, types+'_'+'classification_report_'+self.model_version+'.csv')
        # #logging.debug('Metrics calculation completed')
        
        ### Storing the results/metrics into csv file
        confusion_matrix_report_path = report_path.replace("classification_report","confusion_matrix")

        class_report.to_csv(report_path)
        print(f"[*] Classification report for {type} data saved at {report_path}")
        conf_df.to_csv(confusion_matrix_report_path)
        print(f"[*] Confusion matrix for {type} data saved at {confusion_matrix_report_path}")
        # #logging.debug('Storing the metrics results')
        return class_report, conf_df               

    def model_prediction(self,test_df: pd.DataFrame,process_type: str = 'train') -> pd.DataFrame: #model_version = datetime.now().strftime('%d%m%Y'),
        """
        - Features subset from the dataset for prediction
        - Loading the model objects and encoder objects for model prediction
        - Model prediction on the test dataset and inverse transform to get the labels for the encoder
        - Subset the required columns based on the process type and return the predicted dataframe

        Parameters
        ----------
        test_df (pd.DataFrame): test data with features for model prediction
        process_type (str) - train/inference for getting the prediction

        Returns
        -------

        """
        X_test = test_df.copy()

        # Features subset from the dataset for prediction          
        features_list = self._feature_subset()
            
        
        #logger.debug('Reading the model objects')

        # Model prediction on the test dataset and inverse transform to get the labels for the encoder
        #logger.debug('Started the model prediction on test data')
        predicted_model_prob = self.clf_model.predict_proba(X_test[features_list].values)
        predicted_prob_class = [(pred[np.argmax(pred)],np.argmax(pred)) for pred in predicted_model_prob]
        predicted_prob = [val[0] for val in predicted_prob_class]        
        predicted_label = [self.encoder.inverse_transform([val[1]])[0] for val in predicted_prob_class]
        #logger.debug('Completed the model prediction on test data')
        
        # Storing the model results in predicted results
        #logger.debug('Storing the model prediction results')

        X_test['predicted_datatype_l1'] = predicted_label
        X_test['predicted_probability_l1'] = predicted_prob
             
        # Subset the required columns based on the process type
        if process_type == 'inference':
            test_subset = X_test[['master_id','predicted_datatype_l1','predicted_probability_l1']]            
        else:
            test_subset = X_test[['master_id','grouped_datatype','predicted_datatype_l1','predicted_probability_l1']]

        return test_subset

    def build_ML_Model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_validation: pd.DataFrame, y_validation: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, optimize: bool = False , label_name:str = 'grouped_datatype'):# model_version = datetime.now().strftime('%d%m%Y')):
        """
        - Features subset from the dataset for training
        - Defining the model objects to be used for the model training
        - Creating and storing model on Training Data
        - Predicting and storing the model metrics on test and validation dataset

        Parameters
        ----------
        X_train (pd.DataFrame): Train feature for model training
        y_train (pd.DataFrame): Train label for model training
        X_validation (pd.DataFrame): Validation feature for model training
        y_validation (pd.DataFrame): Validation label for model training
        X_test (pd.DataFrame): Test feature for model training
        y_test (pd.DataFrame): Test label for model training
        optimize (bool): Flag for hyperparameter tune the model
        label_name (str): Target name

        Returns
        -------
        results from the model_metrics() methods such as classification report and confusion matrix
        """
        print(f"[*] Model building started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # Features subset from the dataset for training          
        features_list = self._feature_subset()

        #logger.debug('Building Voting Classifier..')
        
        # Defining the model objects to be used for the model training
        XGB = XGBClassifier()
        rfc = RandomForestClassifier()
        svc = SVC(probability=True)
        
        clf = VotingClassifier(estimators=[('XGB', XGB), ('RF', rfc), ('SVC', svc)], voting='soft', 
                               weights=[MODEL_CONFIG['THRESHOLD']['XGB_MODEL_WEIGHT'],
                                        MODEL_CONFIG['THRESHOLD']['RF_MODEL_WEIGHT'],MODEL_CONFIG['THRESHOLD']['SVM_MODEL_WEIGHT']])

        # Creating the model on Training Data
        clf.fit(X_train[features_list].values,y_train)
        
        # Storing the model objects
        #logger.debug('Writing the XGBoost model objects')
        model_path = os.path.join(self.container_path, self.model_objects_directory,'di_l1_classifier_xgb_'+self.model_version+'.pkl')
        encoder_path = os.path.join(self.container_path,self.model_objects_directory,"di_l1_classifier_encoder_"+self.model_version+'.pkl')

        joblib.dump(clf,model_path,compress=3)

        print(f"[*] Classifier model saved at {model_path} ...")
        # Predicting on the test dataset
        #logger.debug('Prediction on the test data')

        _model = L1Model.for_inference(
            encoder_path=encoder_path,
            model_path=model_path
        )

        X_test_pred = _model.model_prediction(test_df=X_test, process_type='test')

        test_pred_path = os.path.join(self.container_path, self.model_results_directory,f"di_l1_classifier_test_predicted_xgb_{datetime.now().strftime('%d%m%Y')}.csv")
        X_test_pred.to_csv(test_pred_path,index=False)
        print(f"[*] Test predictions saved at {test_pred_path} ...")
        ### Get the model metrics for the test dataset
        #logger.debug('Calculating the model metrics on test dataset')
        class_report_test, conf_df_test = self.model_metrics(actual = X_test_pred[label_name], pred = X_test_pred['predicted_datatype_l1'], type='test')

        ### Predicting on the validation dataset
        #logger.debug('Prediction on the validation data')
        X_validation_pred = _model.model_prediction(test_df=X_validation, process_type='validation')
        val_pred_path = os.path.join(self.container_path, self.model_results_directory,f"di_l1_classifier_validation_predicted_xgb_{datetime.now().strftime('%d%m%Y')}.csv")
        X_validation_pred.to_csv(val_pred_path,index=False)
        print(f"[*] Validations predictions saved at {val_pred_path} ...")

        ### Get the model metrics for the validation dataset
        #logger.debug('Calculating the model metrics on Validation dataset')
        class_report_val, conf_df_val = self.model_metrics(actual = X_validation_pred[label_name],pred = X_validation_pred['predicted_datatype_l1'], type='validation')

        print(f"[*] Model building completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return X_test_pred, class_report_test, conf_df_test, X_validation_pred, class_report_val, conf_df_val
        # #logging.debug('Model building completed')
