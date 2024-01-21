# ====================================================================
#  Importing the required python packages
# ====================================================================

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold

from legoai.core.path_configuration import PATH_CONFIG
from legoai.core.model_configuration import MODEL_CONFIG

# Creating an object
# #logger = #logger.get#logger(
#     parent_folder_name= 'datatype_l1_identification' , child_folder_name='model'
# )
 

container_path = PATH_CONFIG["CONTAINER_PATH"]
 
class L1Model:
    
    ### Setting the path fir the various path for model related info
    def __init__(self):
        self.model_objects_directory = os.path.join(PATH_CONFIG['MODEL_OBJECTS_PATH'], 'datatype_l1_identification')
        self.model_results_directory = os.path.join(PATH_CONFIG['MODEL_RESULTS_PATH'], 'datatype_l1_identification')
        self.model_metrics_directory = os.path.join(PATH_CONFIG['MODEL_METRICS_PATH'], 'datatype_l1_identification')
        self.data_directory = os.path.join(PATH_CONFIG['ANALYT_DATA_PATH'], 'datatype_l1_identification')
        self.label_directory =  os.path.join(PATH_CONFIG['GT_PATH'], 'datatype_l1_identification')
    
    # ====================================================================
    # group_labels: 
    #  - groups the similar datatypes into one single type
    # Parameters: 
    #     datatype- input datatype 
    # ====================================================================     
    def _group_labels(self,datatype: str) -> str:
        
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
        
    # ====================================================================
    # feature_subset: 
    #  - Returns the list of required features for the modeling
    # ====================================================================             
    def _feature_subset(self) -> list:
        
        ##logger.debug('Subsetting the char, par, word, rest features')
        feats_cols = ['dateRatio', 'wordlen_mean', 'rangeRatio', 'floatRatio', 'zero_flag', 'intRatio', 'alphaNumRatio', 'alphaRatio', 'frac_unique_sample','flag_numcells']
        
        # Returns the list of required features for the modeling
        return feats_cols

    # ====================================================================
    # train_test_creation: 
    #  - Reading the required features file and label data from feature and GT path
    #  - Merge the feature and label data to create grouped datatype
    #  - Split the data into train, test and validation for model training
    #  - Encoding the grouped datatype for modeling and store the encoder object
    #  - Return train, test and validation dataset for modeling
    # Parameters: 
    #     label- datatype
    #     model_version- model version    
    # ====================================================================         
    def train_test_creation(self, label: str = 'grouped_datatype',model_version = datetime.now().strftime('%d%m%Y')):
        
        # Reading the required features file from feature path
        #logger.debug('Reading features data')
        feats_path = os.path.join(container_path, self.data_directory,'di_l1_consolidated_feats_data.csv')
        
        # Reading the required label file from GT path
        #logger.debug('Reading label data')
        label_path = os.path.join(container_path, self.label_directory,'di_l1_ground_truth.csv')
        
        # Reading the required feature and label file from path
        model_data = pd.read_csv(feats_path)
        labelled_data = pd.read_csv(label_path)
        
        # Merge the feature and label data to create grouped datatype
        model_label_data = pd.merge(model_data,labelled_data[['master_id','datatype']],on='master_id')
        
        #logger.debug('Grouping datatype')        
        model_label_data['grouped_datatype'] = model_label_data['datatype'].apply(self._group_labels)
        model_label_data['grouped_datatype'] = model_label_data['grouped_datatype'].str.lower()
        
        #logger.debug('Train - Test dataset creation started...')
        
        ### Train/Test Data split
        train_file_name = [MODEL_CONFIG['PREPROCESS_CONSTANT']['TRAIN_DATASET'],MODEL_CONFIG['PREPROCESS_CONSTANT']['VALIDATION_DATASET']]
        
        # Split the data into train, test and validation for model training
        train_val_data = model_label_data[model_label_data['repo_name'].isin(train_file_name)]
        train_data = train_val_data[train_val_data['repo_name']== MODEL_CONFIG['PREPROCESS_CONSTANT']['TRAIN_DATASET']]
        valid_data = train_val_data[train_val_data['repo_name']== MODEL_CONFIG['PREPROCESS_CONSTANT']['VALIDATION_DATASET']]        
        test_data = model_label_data[~model_label_data['repo_name'].isin(train_file_name)]
        
        print('Train: ',train_data.shape, 'Valid: ',valid_data.shape, 'Test: ',test_data.shape)
        
        ### Train - Test - Validation Set
        X_train = train_data
        y_train = train_data[label].values.flatten()
        
        X_validation = valid_data
        y_validation = valid_data[label].values.flatten()
        
        X_test = test_data
        y_test = test_data[label].values.flatten()
        
        ### Encoding the target variables
        num_classes = len(set(y_train))
                            
        encoder = LabelEncoder()
        encoder.fit(y_train)
        
        # Encoding the label data for the modeling 
        y_train = encoder.transform(y_train)
        y_validation = encoder.transform(y_validation)
        y_test = encoder.transform(y_test)
        
        ### Storing the label encoder object
        model_path  = open(os.path.join(container_path, self.model_objects_directory,'di_l1_classifier_encoder_'+model_version+'.pkl'),'wb')
        joblib.dump(encoder, model_path)
        
        #logger.debug('Train - Test dataset creation completed...')

        ### Return train, test and validation dataset for modeling
        return (X_train, y_train, X_validation, y_validation, X_test, y_test)
    
    # ====================================================================
    # optimize_model: 
    #  - Define the possible parameters for the model optimization
    #  - Perform the Grid search for the best model parameter
    #  - Return the best parameters from optimization
    # Parameters: 
    #    X_train - Feature data
    #    y_train - Target data
    # ====================================================================             
    def optimize_model(self,X_train: pd.DataFrame, y_train: np.array, k: int=3) -> dict:
        
        # Parameters for the model optimization
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
       
    # ====================================================================
    # model_metrics: 
    #  - Calculate the model metrics such as classification report and confusion matrix
    #  - Perform the Grid search for the best model parameter
    #  - Return the best parameters from optimization
    # Parameters: 
    #    actual- Actual value of target
    #    pred - Predicted value of target
    #    types - input model name
    #    model_version - model version
    # ====================================================================              
    def model_metrics(self, actual, pred, types = 'di_l1_classifier_test_predicted_xgb',model_version = datetime.now().strftime('%d%m%Y')):
        
        #logging.debug('Metrics calculation started')
        
        # Calculate the classification report for the input actual and predicted
        report = classification_report(actual, pred, output_dict=True)        
        class_report = pd.DataFrame(report).transpose() 
        
        # Calculate the confusion matrix for the input actual and predicted
        conf_mat = confusion_matrix(actual, pred)
        conf_df = pd.DataFrame(conf_mat)
        
        # Path for storing the classification report and confusion matrix report for the input actual and predicted
        report_path = os.path.join(PATH_CONFIG["CONTAINER_PATH"], self.model_metrics_directory, types+'_'+'classification_report_'+model_version+'.csv')
        # #logging.debug('Metrics calculation completed')
        
        ### Storing the results/metrics into csv file 
        class_report.to_csv(report_path)       
        conf_df.to_csv(report_path.replace('classification_report','confusion_matrix'))       
        
        # #logging.debug('Storing the metrics results')
        return class_report, conf_df               
        
    # ====================================================================
    # model_prediction: 
    #  - Features subset from the dataset for prediction 
    #  - Loading the model objects and encoder objects for model prediction
    #  - Model prediction on the test dataset and inverse transform to get the labels for the encoder
    #  - Subset the required columns based on the process type and return the predicted dataframe
    # Parameters: 
    #    test_df - test dataframe with features for model prediction
    #    model_version - model version
    #    process_type - train/inference for getting the prediction    
    # ====================================================================                      
    def model_prediction(self,test_df: pd.DataFrame, model_version = datetime.now().strftime('%d%m%Y'),process_type: str ='train') -> pd.DataFrame:
        print("[*] L1 model prediction started...")
        
        X_test = test_df.copy()
        
        # Features subset from the dataset for prediction          
        features_list = self._feature_subset()
            
        # Loading the model objects for model prediction
        #logger.debug('Reading the model objects')
        model_path = os.path.join(container_path, self.model_objects_directory,'di_l1_classifier_xgb_'+model_version+'.pkl')
        clf_model = joblib.load(model_path)  
        
        # Loading the encoder objects for model prediction
        encoder_path = os.path.join(container_path, self.model_objects_directory,'di_l1_classifier_encoder_'+model_version+'.pkl')
        encoder = joblib.load(encoder_path)
        
        # Model prediction on the test dataset and inverse transform to get the labels for the encoder
        #logger.debug('Started the model prediction on test data')
        predicted_model_prob = clf_model.predict_proba(X_test[features_list].values)
        predicted_prob_class = [(pred[np.argmax(pred)],np.argmax(pred)) for pred in predicted_model_prob]
        predicted_prob = [val[0] for val in predicted_prob_class]        
        predicted_label = [encoder.inverse_transform([val[1]])[0] for val in predicted_prob_class]
        #logger.debug('Completed the model prediction on test data')
        
        # Storing the model results in predicted results
        #logger.debug('Storing the model prediction results')

        X_test['predicted_datatype_l1']= predicted_label
        X_test['predicted_probability_l1']= predicted_prob
             
        # Subset the required columns based on the process type
        if process_type == 'inference':
            test_subset = X_test[['master_id','predicted_datatype_l1','predicted_probability_l1']]            
        else:
            test_subset = X_test[['master_id','grouped_datatype','predicted_datatype_l1','predicted_probability_l1']]
            
        print("[*] L1 model prediction complete...")
        return test_subset
    
    # ====================================================================
    # build_ML_Model: 
    #  - Features subset from the dataset for training
    #  - Defining the model objects to be used for the model training
    #  - Creating and storing model on Training Data
    #  - Predicting and storing the model metrics on test and validation dataset
    # Parameters: 
    #    X_train - Train feature for model training
    #    y_train - Train label for model training
    #    X_validation - Validation feature for model training
    #    y_validation - Validation label for model training
    #    X_test - Test feature for model training
    #    y_test - Test label for model training   
    #    optimize - Flag for hyperparameter tune the model
    #    label_name - Target name   
    #    model_version - model version
    # ====================================================================              
    def build_ML_Model(self,X_train, y_train, X_validation, y_validation, X_test, y_test, optimize=False , label_name='grouped_datatype', model_version = datetime.now().strftime('%d%m%Y')):
        
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
        model_path = os.path.join(container_path, self.model_objects_directory,'di_l1_classifier_xgb_'+model_version+'.pkl')
        joblib.dump(clf,model_path,compress=3)
        
        # Predicting on the test dataset
        #logger.debug('Prediction on the test data')
        X_test_pred = self.model_prediction(test_df=X_test, process_type='test', model_version=model_version)
        X_test_pred.to_csv(os.path.join(container_path,self.model_results_directory,f"di_l1_classifier_test_predicted_xgb_{datetime.now().strftime('%d%m%Y')}.csv"),index=False)
        
        ### Get the model metrics for the test dataset
        #logger.debug('Calculating the model metrics on test dataset')
        class_report, conf_df = self.model_metrics(actual = X_test_pred[label_name], pred = X_test_pred['predicted_datatype_l1'], types='di_l1_classifier_test_predicted_xgb', model_version = model_version)

        ### Predicting on the validation dataset
        #logger.debug('Prediction on the validation data')
        X_validation_pred = self.model_prediction(test_df=X_validation, process_type='validation' ,model_version = model_version)
        X_validation_pred.to_csv(os.path.join(container_path,self.model_results_directory,f"di_l1_classifier_validation_predicted_xgb_{datetime.now().strftime('%d%m%Y')}.csv"),index=False)
        
        ### Get the model metrics for the validation dataset
        #logger.debug('Calculating the model metrics on Validation dataset')
        class_report, conf_df = self.model_metrics(actual = X_validation_pred[label_name],pred = X_validation_pred['predicted_datatype_l1'], types='di_l1_classifier_validation_predicted_xgb', model_version = model_version)

        
        # #logging.debug('Model building completed')