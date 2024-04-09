<p align="center">
      <img src="../../../documentation/gray%20LEG0AI%20Cymk%20color%20JPG.jpg" alt="LegoAI Logo" width=350>
  <h3 align="center"><i>Empowering Business Users With Self Serve Analytics</i></h3>
  
## Table of contents
- [Technical Details](#details)
- [Configuration](#configuration)
- [Usage](#usage)
  
## Details
> [!NOTE]
>  ### Feature Specific Terminologies
>   - **repo/dataset**: directory or folder that consits of all tables.
>   - **table**: tables are files (csv or other extension) that is present with in the repo.
>   - **column**: individual columns within the table.
>   - ### Structure
>      ```
>     LEGO AI/DATA
>     └───ecommerce_data ( repo )
>         └───olist_customers_dataset.csv ( table )
>             └─── customer_id (column)
>             └─── customer_name (column)
>             └─── customer_address (column)
>     ```
>   - **master_id**: combination of repo, table and column name, i.e. formed as **repo$$##$$table$$##$$column** to uniquely identify each row used during feature processing and model building.

## Configuration
  ### Model configuration
  All the configuration needed for inference and training is stored in legoai/config.yaml and you can define your own configuration in **config.yaml**. (**specific to both training and inference**):
  ``` 
    PREPROCESS_CONSTANT:
        NUMBER_PATTERN: '[0-9]'
        TEXT_PATTERN: '[a-zA-Z]'
        WORD_PATTERN: '[\w+]'
        SPECIAL_CHAR_PATTERN: '[!@#$%^&*(),?":;{}\''|<>~`=_\\/+-]'
        RANGE_PATTERN: '(\d+)\s*[-|to]+\s*(\d+)' 
        URL_PATTERN: 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    THRESHOLD:
        CORPUS_WORD_LIMIT: 3 # minimum length of the word to create words list for spell check or valid text ( used in lexical matching of certain values with english dictionary words)
        PARTITION_SIZE: 100
        DATE_MIN_YEAR: 1800 # minimum range value for date extraction
        DATE_MAX_YEAR: 2100 # maximum range value for date extraction
        SYMSPELL_PREFIX_LEN: 7 # symspell related configuration ( symspell is used for retrieving cleaned name ( column name, table name ) ) 
        SYMSPELL_EDIT_DIST: 0 # symspell related configuration
        SYMSPELL_TERM_INDEX: 0 # symspell related configuration
        SYMSPELL_COUNT_INDEX: 1 # symspell related configuration
        DATA_VALUES_LIMIT: 1000000 # no of samples or column values used for feature creation
        XGB_MODEL_WEIGHT: 0.4 # for voting classifier
        RF_MODEL_WEIGHT: 0.3 # for voting classifier
        SVM_MODEL_WEIGHT: 0.3 # for voting classifier
    
    L3PARAMS:
        SAMPLE_SIZE: 100 #  size to sample values from a column and identify date time 
        DF_SAMPLE_SIZE: 10 # size to sample values from a column and use for prompt creation in identifying dimension and measure
    
    FILE_TYPE:
        FILE_ENCODING: 'iso-8859-1'
        DTYPE_ENCODING: 'unicode'
  ```
  ### Path Configuration
  All the configuration needed for saving intermediate results, reading the results is stored in **.env** ( environmet variables) and you can define your own **.env** file. (**specific for training**)
```
  CONTAINER_PATH = data/Lego_AI # Base folder of the entire folder structure
  INT_DATA_PATH = intermediate/ # Path for storing the intermediate files during feature creation
  ANALYTICAL_DATA_PATH = analytical_data/ # Path for storing the final features used for modeling
  MODEL_METRICS_PATH = model/model_metrics # Storing classification and confusion matrix report after training (for test and validation dataset)
  MODEL_OBJECTS_PATH = model/model_objects # Storing encoder and classifier model
  MODEL_RESULTS_PATH = model/model_results # Storing final results after model training (for test and validataion dataset)
```
  _The folder structure will look as follow:_
  ```
    DATA/LEGO_AI ( CONTAINER_PATH )
    +---analytical_data ( ANALYTICAL_DATA_PATH)
    |   \---datatype_l1_identification
    |           di_l1_consolidated_feats_data.csv
    |
    +---intermediate ( INT_DATA_PATH )
    |   \---datatype_l1_identification
    |           db_extract_feats.csv
    |           real_world_dataset_feats.csv
    |           web_crawl_dataset_feats.csv
    |
    \---model ( MODEL PATH ) 
        +---model_metrics ( MODEL_METRICS_PATH )
        |   \---datatype_l1_identification
        |           di_l1_classifier_test_predicted_xgb_classification_report_13052023.csv ( classification report  test data )
        |           di_l1_classifier_test_predicted_xgb_confusion_matrix_13052023.csv ( confusion matrix report test data )
        |           di_l1_classifier_validation_predicted_xgb_classification_report_13052023.csv ( classification report validation data )
        |           di_l1_classifier_validation_predicted_xgb_confusion_matrix_13052023.csv ( confusion matrix report validation data )
        |
        +---model_objects ( MODEL_OBJECTS_PATH )
        |   \---datatype_l1_identification
        |           di_l1_classifier_encoder_13052023.pkl ( final encoder )
        |           di_l1_classifier_xgb_13052023.pkl ( final trained classifier model )
        |
        \---model_results ( MODEL_RESULTS_PATH )
            \---datatype_l1_identification
                    di_l1_classifier_test_predicted_xgb_12032024.csv ( prediction result test data )
                    di_l1_classifier_validation_predicted_xgb_12032024.csv ( prediction result validation data )
  ```
## Usage
  ### Inference
  - ```legoai.DataTypeIdentificationPipeline.prediction_pipeline```  
> [!Note]
> - Returns an datatype identification final result from L1 model and L2 model (if specified).   
 - ```Parameters```
   - ```input_path (str): the directory or file path to the inference dataset```
   - ```output_path (str): output path to save all the results, i.e. processed files,features, and final output```
   - ```L2_predict (bool): flag to denote whether or not to run L2 model ( default is false )```
- ```Returns```
   - ```dataframe with L1 prediction & L2 prediction ( if L2_predict = True )```
```python
  - Example ( encoder path and model path taken reference from folder structure in Path configuration above )
       
         di_result = DataTypeIdentificationPipeline.pretrained_pipeline(
            input_path = "LegoAI/data/ecommerce_data",
            output_path = "di_opensource",
            L2_predict = True
         )  
   ```
### Training
- ```legoai.DatatypeIdentification.train```
> [!NOTE]
> - Executes the training pipeline
  - ```Parameters```
    - ```input_path (str) (required): raw dataset path for training ```
    - ```gt_path (str) (required):ground truth path for the datatype for all the columns```
    - ```output_path (str (optional): final output path to save all models, features, and reports (default will be 'datatype_identification_training/')```
 - ```Returns```
    - ```classification and confustion matrix report (dataframe) and prediction result of test and validation dataset```
```python
- Example
      di = DataTypeIdentificationPipeline()

      # provide data path for training and its corresponding ground truth or labels
      dataset_path = r"Lego AI/DI_OPENSOURCE/data/Lego_AI/input/raw_data"
      ground_truth_path = r"Lego AI/DI_OPENSOURCE/data/Lego_AI/input/ground_truth"
     # final output path to save intermediate files, classification and confusion matrix reports , and encoder and classifier model. can be defined in .env file under "CONTAINER_PATH" 
    output_path = r"datatype_identification_training"

    # training
    di.train(input_path=dataset_path,
            gt_path=ground_truth_path,
             output_path= output_path
      )
```
>[!IMPORTANT]
> Sucessfull training outputs **model_version** which can be used in the inference by providing the model_version the model_version is created by default using current year,month and day.
  
> [!NOTE]
> For full example with output see below.  

_**Training Input Path Structure**_
```
LEGO AI/DI_OPENSOURCE/DATA/LEGO_AI/INPUT/RAW_DATA
└───datatype_l1_identification
    ├───db_extract
    │       db_extract.json
    │
    ├───real_world_dataset
    │       real_world_dataset.json
    │
    └───web_crawl_dataset
            web_crawl_dataset.json
```
> [!NOTE]
> for example data refer to this file [real_world_dataset](../../../training_directory/Lego_AI/input/raw_data/datatype_l1_identification/real_world_dataset)

_**Training Ground Truth (Label) Path Structure**_
```
LEGO AI/DI_OPENSOURCE/DATA/LEGO_AI/INPUT/GROUND_TRUTH
└───datatype_l1_identification
        di_l1_ground_truth.csv
```
> [!NOTE]
> for example data refer to this file [ground_truth](../../../training_directory/Lego_AI/input/ground_truth/datatype_l1_identification)
 
> [!IMPORTANT]
> **master_id** column is required in ground truth for merging final data for model building process


  
     
