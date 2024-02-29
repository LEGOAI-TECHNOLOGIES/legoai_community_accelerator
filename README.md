<p align="center">
  <a href="https://github.com/legoai-opensource/legoai">
      <img src="https://github.com/narotsitkarki/DI_OPENSOURCE/assets/154975391/d3f48afd-9b05-4828-8115-8795fac9f308" alt="LegoAI Logo" width=180>
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
      ✅ **Integer**  
      ✅ **Float**  
      ✅ **Alphanumeric**  
      ✅ **open ended text**  
      ✅ **close ended text**  
      ✅ **date & time**  
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
- _**Inference Example**_
  ``` 
  from legoai import DataTypeIdentificationPipeline

  # provide dataset path and openai key
  dataset_path = "D:\LegoAI\data\ecommerce_data
  api_key = "your-api-key"

  # load the pretrained pipeline
  di_pipeline = DataTypeIdentificationPipeline.pretrained_pipeline(openai_api_key = api_key)

  # provide the dataset and final output save path, also final result dataframe is returned
  result = di_pipeline.predict(dataset_path = dataset_path, save_to = "di_output.csv")

  print(result.head())
  ```




