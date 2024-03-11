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
- [Examples](#examples)

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
  - _currently uses trained XGBClassifier model._
  
- [L3 model](https://github.com/narotsitkarki/DI_OPENSOURCE/blob/master/legoai/modules/datatype_identification/l3_model.py)
  - _This 2nd part classifies the column into one level deep and further classifies l1 identified datatypes, specifically float and integer into dimension or measure, and         also classifies date and time into certain format of date and time such as YYYY/mm/dd or YYYY-mm-dd H:m:s others [see](https://github.com/narotsitkarki/DI_OPENSOURCE/blob/master/legoai/modules/datatype_identification/l3_model.py). other than integer , float and date & time others are kept   same._
  - llm is used for this
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

## Examples  
_**Inference Example**_
``` 
from legoai import DataTypeIdentificationPipeline

# provide input path and openai key
input_path = "D:\LegoAI\data\ecommerce_data"
api_key = "your-openai-api-key"

# load the pretrained pipeline
di_pipeline = DataTypeIdentificationPipeline.pretrained_pipeline(openai_api_key = api_key)

# provide the dataset and final output save path, also final result dataframe is returned
result = di_pipeline.predict(dataset_path = dataset_path, save_to = "di_output.csv")

print(result.head())
```
  _**Inference Example Output**_
  ``` 
  [*] MODEL VERSION: 13052023  
  [*] processed product_category_name_translation.csv: 100%|██████████| 9/9 [00:11<00:00,  1.30s/it]  
  [*] total columns present in the ecommerce_data 52  
  [*] Feature Extraction Started... 2024-02-29 15:35:21  
  [*] Feature Creation Finished. Processed 52 rows in 0:01:43.627949  
  [*] Inference complete ... took 2.29 minute ...  
             column_name_clean predicted_datatype_l1 predicted_datatype_l3    
   0               customer id          alphanumeric          alphanumeric  
   1        customer unique id          alphanumeric          alphanumeric  
   2  customer zip code prefix               integer     integer_dimension  
   3             customer city      close_ended_text      close_ended_text  
   4            customer state      close_ended_text      close_ended_text  
```
_**Training Example**_
```
di = DataTypeIdentificationPipeline()  

# provide data path for training and its corresponding ground truth or labels  
dataset_path = "D:\Lego AI\DI_OPENSOURCE\data\Lego_AI\input\\raw_data"  
ground_truth_path = "D:\Lego AI\DI_OPENSOURCE\data\Lego_AI\input\ground_truth"  

#give model version to save the final encoders and classifier model under the given version  
model_version = "13052023"  
di.train(dataset_path=dataset_path, gt_path=ground_truth_path, model_version=model_version)  
```
  _**Training Example Output**_
``` 
[*] Extracting files meta information ...    
0%|          | 0/3 [00:00<?, ?it/s]  
[*] Meta Data Row Count db_extract.json:  (362, 13)  
[*] Feature Extraction Started... 2024-02-29 15:17:29  
[*] Feature Creation Finished. Processed 362 rows in 0:03:08.564266  
33%|███▎      | 1/3 [03:11<06:23, 191.52s/it]   
[*] Meta Data Row Count real_world_dataset.json:  (36375, 7)  
[*] Feature Extraction Started... 2024-02-29 15:20:39  
[*] Feature Creation Finished. Processed 36375 rows in 0:01:11.574827  
67%|██████▋   | 2/3 [04:24<02:01, 121.96s/it]  
[*] Meta Data Row Count web_crawl_dataset.json:  (3585, 7)  
[*] Feature Extraction Started... 2024-02-29 15:21:51    
[*] Feature Creation Finished. Processed 3585 rows in 0:00:12.684016
100%|██████████| 3/3 [04:37<00:00, 92.58s/it]    
[*] Combining all features into single file ...    
[*] Consolidated features saved at data\Lego_AI\analytical_data\datatype_l1_identification\di_l1_consolidated_feats_data.csv ...  
[*] MODEL VERSION: 13052023  
[*] Features: (49794, 1714) , Labels: (49794, 2)  
[*] Final Merged Features and Labels: (49794, 1715)  
[*] Train:  (36375, 1716) Valid:  (3585, 1716) Test:  (9834, 1716)  
[*] Label encoder saved at data\Lego_AI\model\model_objects\datatype_l1_identification\di_l1_classifier_encoder_13052023.pkl ...  
[*] Model building started at 2024-02-29 15:48:32  
[*] Classifier model saved at data\Lego_AI\model\model_objects\datatype_l1_identification\di_l1_classifier_xgb_13052023.pkl ...  
[*] Test predictions saved at data\Lego_AI\model\model_results\datatype_l1_identification\di_l1_classifier_test_predicted_xgb_29022024.csv ...  
[*] Validations predictions saved at data\Lego_AI\model\model_results\datatype_l1_identification\di_l1_classifier_validation_predicted_xgb_29022024.csv ...  
[*] Model building completed at 2024-02-29 15:49:11  
```





