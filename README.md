<p align="center">
  <a href="https://github.com/legoai-opensource/legoai">
      <img src="https://github.com/narotsitkarki/DI_OPENSOURCE/blob/master/documentation/gray%20LEG0AI%20Cymk%20color%20JPG.jpg" alt="LegoAI Logo" width=350>
  </a>

  <h3 align="center"><i>Empowering Business Users With Self Serve Analytics</i></h3> 

## What is LegoAI ?
_**GenAI** Powered analytics platform that automatically converts business requirements into executable machine language, enabling users with explainable & accurate insights at unparalleled speed & scale._
## Table of contents
- [Main Features](#main-features)
- [Datatype Identification](#datatype-identification)
- [Where to get it](#where-to-get-it)
- [License](#license)
- [Documentation](#documentation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)

## Main Features
The project has different pipelines ... currently only released pipeline is:
- [Datatype Identification](https://github.com/narotsitkarki/DI_OPENSOURCE/tree/master/legoai/modules/datatype_identification)  <<[**details**](#datatype-identification)>>
- coming soon ...
## Datatype Identification
_As simple as it sounds this pipeline helps in identifying the datype of all columns within a dataset. The identification is divided into 2 seperate parts_
- [L1 model](https://github.com/narotsitkarki/DI_OPENSOURCE/blob/master/legoai/modules/datatype_identification/l1_model.py)
    - _This 1st part classifies the column into one of the 7 datatypes_:  
      ✅ **integer**  
      ✅ **float**  
      ✅ **alphanumeric**  
      ✅ **open_ended_text**  
      ✅ **close_ended_text**  
      ✅ **date & time**  
      ✅ **range_type**  
      ✅ **others ( if not found any)**  
  - _currently uses VotingClassifier between (XGBClassifier, RandomForest and SVC) with soft voting techique._
  
- [L3 model](https://github.com/narotsitkarki/DI_OPENSOURCE/blob/master/legoai/modules/datatype_identification/l3_model.py)
  - _This 2nd part classifies the column into one level deep and further classifies l1 identified datatypes, specifically float and integer into dimension or measure, and         also classifies date and time into certain format of date and time such as YYYY/mm/dd or YYYY-mm-dd H:m:s others [see](https://github.com/narotsitkarki/DI_OPENSOURCE/blob/master/legoai/modules/datatype_identification/l3_model.py). other than integer , float and date & time others are kept   same._
  - _LLM is used for this._
> [!IMPORTANT]  
> L3 model requires OpenAI API key.
    
## Where to get it?
The source code can be found at https://github.com/legoai-opensource/legoai
Binary installers for the latest released version are available at the [Python Package Index (PyPI)](https://pypi.org/project/)

```
# PyPI
> pip install legoai
```

## License

## Documentation
> [!NOTE]
>  ### Feature Specific Terminologies
>   - **repo**: _directory or folder that consits of all tables_.
>   - **table**: _tables are files (csv or other extension) that is present with in the repo_.
>   - **column**: _individual columns within the table_.
>   - ### Structure
>      ```
>     D:\LEGO AI\DATA
>     └───ecommerce_data ( repo )
>         └───olist_customers_dataset.csv ( table )
>             └─── customer_id (column)
>             └─── customer_name (column)
>             └─── customer_address (column)
>     ```
>   - **master_id**: _combination of repo, table and column name, i.e. formed as **repo$$##$$table$$##$$column** to uniquely identify each row used during feature processing and model building_.
    
  ### How does it works ?
  1. **L1 - Datatype Identification - Data Preparation**  
        - Data Preparation includes: ( **for training** )
          - Reading the files from Raw data folder from each repo and store it in dataframe.
          - Check if all the required columns such as ID, Column Name, Table Name, Repo Name are present.
          - If present then use the existing, else create a new ID creation.
          - Create a master id >> Unique ID for each column (Concatenation of Repo Name/Table Name/Column Name).
          - Pass the schema processed dataframe for preprocessing and feature creation.
        - Preprocessing includes:  ( **for training and inference** )
          - Removing ASCII characters.
          - Ignoring certain words from adding to the list.
          - Removing None, NaN, Blanks from the data list.
          - Remove the data if it contains only punctuations.
          - Convert the data to string format.
          - Replacing extra whitespaces from both start and end.  
      - Feature creation includes:  ( **for training and inference** )
          - Post preprocessing, the data is passed for feature creation.
          - Feature creation includes character level statistical metrics, and custom features.
 
  2. **L1 - Datatype Identification - Model Building**
     - Create train, test, and validation data obtained from feature creation process ( 60%-20%-20% ) rule.
     - Fit and save encoder for the label ( i.e. datatype).
     - Build VotingClassifier model ( i.e soft voting technique between XGBClassifier, RandomForestClassifier, and SVC).
     - Save the final classifier model, generate and store prediction result, classification and confustion matrix report for test and validation set.
    
  3. **L1 - Datatype Identification - Inference**
     - Preprocessing and Feature creation process is same as above.
     - Load the encoder and classifier model.
     - Run prediction through the model.
    
  4. **L3 - Datatype Identification - Inference**
     - Uses results obtained from L1 inferencing.
     - Creates LLM prompt form integer and float datatype and runs prompt, rule based approach for identifying appropriate date & time format for date & time datatype.
     - Save final L1 and L2 prediction.
     
    
    
## Configuration
  ### Model configuration
  _All the configuration needed for inference and training is stored in legoai/config.yaml and you can define your own configuartion in **config.yaml**. (**specific to both training and inference**):_
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
  _All the configuration needed for saving intermediate results, reading the results is stored in **.env** ( environmet variables) and you can define your own **.env** file. (**specific for training**)_
```
  CONTAINER_PATH = data\Lego_AI # Base folder of the entire folder structure
  INT_DATA_PATH = intermediate # Path for storing the intermediate files during feature creation
  ANALYTICAL_DATA_PATH = analytical_data # Path for storing the final features used for modeling
  MODEL_METRICS_PATH = model\model_metrics # Storing classification and confusion matrix report after training (for test and validation dataset)
  MODEL_OBJECTS_PATH = model\model_objects # Storing encoder and classifier model
  MODEL_RESULTS_PATH = model\model_results # Storing final results after model training (for test and validataion dataset)
```
  _The folder structure looks like_
  ```
    D:\DATA\LEGO_AI ( CONTAINER_PATH )
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
  #### Loading pretrained model  
  - ```legoai.DataTypeIdentificationPipeline.pretrained_pipeline```  
> [!Note]
> - Returns an object with preloaded L1 model and pre instantiated L3 model.  
> - If openai api key not given only instantiates with L1 model.  
> - Encoder and model path can be given to run custom model, else default settings will be used to load pretrained model and encoder.  
 - ```Parameters```
   - ```openai_api_key (str) (optional): an openai api key for L3 model```
   - ```encoder_path (str) (optional): full path to the encoder.```
   - ```model_path (str) (optional): full path to the classifier.```
- ```Returns```
   - ```DatatypeIdentification object with L1 and L3 model loaded.```
```
  - Example ( encoder path and model path taken reference from folder structure in Path configuration above )
       
         di_pipeline = DataTypeIdentificationPipeline.pretrained_pipeline(
           openai_api_key = "your-openai-api-key"
           encoder_path="LegoAI\di_l1_classifier_encoder_13052023.pkl",
           model_path = "Lego_AI\di_l1_classifier_xgb_13052023.pkl"
         )  
   ```
#### Running inference pipeline
  - ```legoai.DataTypeIdentification.predict```  
> [!NOTE] 
> - Executes the inference pipelines and saves the result and also returns the final result
  - ```Parameters```
    - ```input_path (str): the path to the inference dataset```
    - ```output_path (str): output path to save all the results, i.e. processed files,features, and final l1 and l3 model output```
- ```Returns```
  - ```final L1 and L3 (if openai key given) prediction, with generated features dataframe```

```
  - Example
      di_pipeline = DataTypeIdentification.pretrained_pipeline(openai_api_key = "your-openai-api-key")
      di_pipeline.predict(
            input_path = "D:\LegoAI\data\ecommerce_data",
            output_path = "D:\di_opensource"
        )
```
### Training
- ```legoai.DatatypeIdentification.train```
> [!NOTE]
> - Executes the training pipeline
  - ```Parameters```
    - ```input_path (str) (required): raw dataset path for training ```
    - ```gt_path (str) (required):ground truth path for the datatype for all the columns```
    - ```model_version (str) (optional): model version to save under for trained finalized model i.e.(encoder, and classifier) (default current date)```
    - ```output_path (str (optional): final output path to save all models, features, and reports (default will be 'datatype_identification_training\')```
 - ```Returns```
    - ```classification and confustion matrix report (dataframe) and prediction result of test and validation dataset```

```
- Example
      di = DataTypeIdentificationPipeline()

      # provide data path for training and its corresponding ground truth or labels
      dataset_path = r"D:\Lego AI\DI_OPENSOURCE\data\Lego_AI\input\raw_data"
      ground_truth_path = r"D:\Lego AI\DI_OPENSOURCE\data\Lego_AI\input\ground_truth"
     # final output path to save intermediate files, classification and confusion matrix reports , and encoder and classifier model. can be defined in .env file under "CONTAINER_PATH" 
    output_path = r"D:\datatype_identification_training"

    # give model version to save the final encoders and classifier model under the given version
    model_version = "13052023"

    # training
    di.train(input_path=dataset_path,gt_path=ground_truth_path,
             output_path= output_path,
             model_version=model_version
      )
```

> [!NOTE]
> For further clarification see examples below

## Examples  
_**Inference Example**_
``` 
from legoai import DataTypeIdentificationPipeline

# provide input path,output path and openai key
input_path = "D:\LegoAI\data\ecommerce_data"
output_path = "D:\di_opensource"
api_key = "your-openai-api-key"

# load the pretrained pipeline
di_pipeline = DataTypeIdentificationPipeline.pretrained_pipeline(openai_api_key = api_key)

# provide the input path and output path, also final result dataframe is returned
result = di_pipeline.predict(input_path = input_path, output_path = output_path)

print(result.head())
```
_**Input path structure**_
```
D:\LEGO AI\DATA
└───ecommerce_data
        olist_customers_dataset.csv
        olist_orders_dataset.csv
        olist_order_items_dataset.csv
        olist_order_payments_dataset.csv
        olist_order_reviews_dataset.csv
        olist_products_dataset.csv
        olist_sellers_dataset.csv
        product_category_name_translation.csv
```
  _**Inference Example Output**_
  ```   
  [*] processed product_category_name_translation.csv: 100%|██████████| 9/9 [00:11<00:00,  1.30s/it]
  [*] Processed files saved at D:\di_opensource\ecommerce_data_processed  
  [*] Total columns present in the ecommerce_data 52  
  [*] Feature Extraction Started... 2024-02-29 15:35:21  
  [*] Feature Creation Finished. Processed 52 rows in 0:01:43.627949
  [*] Features saved at D:\di_opensource\features\intermediate_features_12032024.csv  
  [*] Inference complete ... took 2.29 minute ...
  [*] Final output saved at D:\di_opensource\output\di_final_output_12032024.csv
             column_name_clean ... predicted_datatype_l1 predicted_datatype_l3    
   0               customer id ...          alphanumeric          alphanumeric  
   1        customer unique id ...          alphanumeric          alphanumeric  
   2  customer zip code prefix ...               integer     integer_dimension  
   3             customer city ...      close_ended_text      close_ended_text  
   4            customer state ...      close_ended_text      close_ended_text  
```
_**Training Example**_
```
from legoai import DataTypeIdentificationPipeline

di = DataTypeIdentificationPipeline()

# provide data path for training and its corresponding ground truth or labels, and the output path to save all the features, model, encoder, classification and confusion report.  
input_path = "D:\Lego AI\DI_OPENSOURCE\data\Lego_AI\input\\raw_data"  
ground_truth_path = "D:\Lego AI\DI_OPENSOURCE\data\Lego_AI\input\ground_truth"
output_path = "D:\data\Lego_AI" # optional , can be defined in .env file under "CONTAINER_PATH" 

#give model version to save the final encoders and classifier model under the given version  
model_version = "13052023"  
di.train( input_path=inputs_path,
          gt_path=ground_truth_path,
          output_path=output_path,
          model_version=model_version)  
```
  _**Training Example Output**_
```
[*] Extracting files meta information ...
  0%|          | 0/3 [00:00<?, ?it/s]
[*] Meta Data Row Count db_extract.json:  (362, 13)
[*] Feature Extraction Started... 2024-03-13 20:05:27
[*] Feature Creation Finished. Processed 362 rows in 0:03:07.700495
[*] Meta information saved at D:\data\Lego_AI\intermediate\datatype_l1_identification\db_extract_feats.csv...
 33%|███▎      | 1/3 [03:11<06:22, 191.23s/it]
[*] Meta Data Row Count real_world_dataset.json:  (36375, 7)
[*] Feature Extraction Started... 2024-03-13 20:08:36
[*] Feature Creation Finished. Processed 36375 rows in 0:01:09.008878
 67%|██████▋   | 2/3 [04:21<02:00, 120.29s/it][*] Meta information saved at D:\data\Lego_AI\intermediate\datatype_l1_identification\real_world_dataset_feats.csv...
[*] Meta Data Row Count web_crawl_dataset.json:  (3585, 7)
[*] Feature Extraction Started... 2024-03-13 20:09:46
[*] Feature Creation Finished. Processed 3585 rows in 0:00:12.406904
100%|██████████| 3/3 [04:34<00:00, 91.50s/it]
[*] Meta information saved at D:\data\Lego_AI\intermediate\datatype_l1_identification\web_crawl_dataset_feats.csv...
[*] Combining all features into single file ...
[*] Consolidated features saved at D:\data\Lego_AI\analytical_data\datatype_l1_identification\di_l1_consolidated_feats_data.csv
[*] MODEL VERSION: 13052023
[*] Features: (40322, 23) , Labels: (49794, 2)
[*] Final Merged Features and Labels: (40322, 24)
[*] Train:  (24193, 25) Valid:  (8064, 25) Test:  (8065, 25)
[*] Encoder saved at D:\data\Lego_AI\model\model_objects\datatype_l1_identification\di_l1_classifier_encoder_13052023.pkl ...
[*] Model building started at 2024-03-13 20:10:00
[*] Classifier model saved at D:\data\Lego_AI\model\model_objects\datatype_l1_identification\di_l1_classifier_xgb_13052023.pkl ...
[*] Test predictions saved at D:\data\Lego_AI\model\model_results\datatype_l1_identification\di_l1_classifier_test_predicted_xgb_13032024.csv ...
[*] Classification report for test data saved at D:\data\Lego_AI\model\model_metrics\datatype_l1_identification\di_l1_classifier_test_predicted_xgb_classification_report_13052023.csv
[*] Confusion matrix for test data saved at D:\data\Lego_AI\model\model_metrics\datatype_l1_identification\di_l1_classifier_test_predicted_xgb_confusion_matrix_13052023.csv
[*] Validations predictions saved at D:\data\Lego_AI\model\model_results\datatype_l1_identification\di_l1_classifier_validation_predicted_xgb_13032024.csv ...
[*] Classification report for validation data saved at D:\data\Lego_AI\model\model_metrics\datatype_l1_identification\di_l1_classifier_validation_predicted_xgb_classification_report_13052023.csv
[*] Confusion matrix for validation data saved at D:\data\Lego_AI\model\model_metrics\datatype_l1_identification\di_l1_classifier_validation_predicted_xgb_confusion_matrix_13052023.csv
[*] Model building completed at 2024-03-13 20:10:24
```
_**Training Input Path Structure**_
```
D:\LEGO AI\DI_OPENSOURCE\DATA\LEGO_AI\INPUT\RAW_DATA
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
> for example data refer to this file [real_world_dataset](https://github.com/narotsitkarki/DI_OPENSOURCE/tree/master/training_directory/Lego_AI/input/raw_data/datatype_l1_identification/real_world_dataset)


_**Training Ground Truth (Label) Path Structure**_
```
D:\LEGO AI\DI_OPENSOURCE\DATA\LEGO_AI\INPUT\GROUND_TRUTH
└───datatype_l1_identification
        di_l1_ground_truth.csv
```
> [!NOTE]
> for example data refer to this file [ground_truth](https://github.com/narotsitkarki/DI_OPENSOURCE/tree/master/training_directory/Lego_AI/input/ground_truth/datatype_l1_identification)
 
> [!IMPORTANT]
> **master_id** column is required in ground truth for merging final data for model building process


## Contributing
Any contributions to this project is welcomed, you can follow the steps below for contribution:
1. Fork the repository.
2. Create a new branch (git checkout -b feature)
3. Make your changes.
4. Commit your changes (git commit -am 'Add new feature')
5. Push to the branch (git push origin feature)
6. Create a new Pull Request.




