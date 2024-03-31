<p align="center">
      <img src="documentation/gray%20LEG0AI%20Cymk%20color%20JPG.jpg" alt="LegoAI Logo" width="350">
 <h3 align="center"><i>Empowering Business Users With Self Serve Analytics</i></h3> 

## What is it ?
This is a cutting-edge project leveraging advanced Machine Learning technologies to accurately discern and classify data types from various values. Designed to enhance data preprocessing and analysis pipelines, this tool automates the often tedious and error-prone task of manually identifying data types.

## Table of contents
- [Main Features](#main-features)
- [Where to get it](#where-to-get-it)
- [Getting Started](#getting-started)
- [Performance](#performance)
- [License](#license)
- [Documentation](#documentation)
- [Contributing](#contributing)
## Main Features
- **Data Type Identification**
    > - Classifying datatypes from various values easily through trained ML model.
    > - Has two models, **L1 model** (_uses Classifier_) that identifies normal datatypes  ( **integer, float, alphanumeric, range_type, date & time, open_ended_text, close_ended_text**)
    > - **L2 model** (_uses LLM_) further classifies L1 datatype result that are **integer** or **float** to measure,dimension or unknown( if not classified) and date & time into one of **41** date-time formats like (YYYY:mm:dd H:m:s), and others ( **alphanumeric, range_type, open_ended_text, close_ended_text** ) remains the same.
> #### L1 and L2 model Classification
<img src="documentation/L1_&_L2_Model_Output.png" alt="L1 and L2 Model" >

  > #### Datatype Identification Inference Workflow  
<img src="documentation/DI_Inference_WorkFlow.png" alt="DI Inference Workflow">

## Where to get it?
Binary installers for the latest released version are available at the [Python Package Index (PyPI)](https://pypi.org/project/)
```
# PyPI
> pip install legoai
```

##  Getting Started
To quickly start using the pipeline just [install](#where-to-get-it) and follow notebook below.
### Datatype Identification ( Inference )
[Inference Notebook](DataTypeIdentification-Inference.ipynb)  
> [!IMPORTANT]
> **openai_api_key** is required if you want to use L2 model, else leave the field then inference will only run for L1 model


## Performance
### Data Type Identification
> [!NOTE]
> **Source Ecommerce:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce  
>  ```Total Tables: 9``` , ```Total Columns: 52```   
> **Source Healthcare:** https://mitre.box.com/shared/static/aw9po06ypfb9hrau4jamtvtz0e5ziucz.zip  
> ```Total Tables: 18```, ```Total Columns: 249``` 

#### Classification Report ( L1 Model )
<img src="documentation/L1_Model_Metrics.png" alt="L1 Model Classification Metrics" width="700">    
  
#### Classification Report ( L2 Model )
<img src="documentation/L2_Model_Metrics.png" alt="L2 Model Classification Metrics" width="700">   
  
#### Execution Chart ( Google Collab Environment )
<img src="documentation/Metrics Google Collab.png" alt="DI Execution Chart" width="700">



## License

## Documentation
For detailed documentation refer to the following links.
- [Data Type Identification](legoai/modules/datatype_identification)

    
    
## Contributing
Any contributions to this project is welcomed, you can follow the steps below for contribution:
1. Fork the repository.
2. Create a new branch feature/* (git checkout -b feature)
3. Make your changes.
4. Commit your changes (git commit -am 'Add new feature')
5. Push to the branch (git push origin feature)
6. Create a new Pull Request.




