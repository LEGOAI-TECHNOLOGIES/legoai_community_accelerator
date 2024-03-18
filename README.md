<p align="center">
  <a href="https://github.com/legoai-opensource/legoai">
      <img src="https://github.com/narotsitkarki/DI_OPENSOURCE/blob/master/documentation/gray%20LEG0AI%20Cymk%20color%20JPG.jpg" alt="LegoAI Logo" width=350>
  </a>

  <h3 align="center"><i>Empowering Business Users With Self Serve Analytics</i></h3> 

## What is it ?
This is a cutting-edge project leveraging advanced Machine Learning technologies to accurately discern and classify data types from various values. Designed to enhance data preprocessing and analysis pipelines, this tool automates the often tedious and error-prone task of manually identifying data types.

## Table of contents
- [Main Features](#main-features)
- [Getting Started](#getting-started)
- [Datatype Identification](#datatype-identification)
- [Where to get it](#where-to-get-it)
- [License](#license)
- [Documentation](#documentation)
- [Contributing](#contributing)
## Main Features
The project has different pipelines ... currently only released module is:
- **Data Type Identification**
    > Classifying datatypes from various values easily through trained ML model.
- coming soon ...  
##  Getting Started
To quickly start using the pipeline just [install](#where-to-get-it) and follow code below.
### Datatype Identification
```python
from legoai.pipeline import DataTypeIdentificationPipeline
di = DataTypeIdentificationPipeline.pretrained_pipeline(
  openai_api_key=<YOUR_OPENAPI_API_KEY>
)
result = di.predict(
  input_path=<DIRECTORY PATH OF THE DATASET> ,
  output_path=<DIRECTORY PATH TO STORE GENERATED FEATURES AND FINAL RESULT>
)
print(result.head())
```
 #### Sample Input path
 ```
input_path = LEGOAI/DATA/ecommerce_data

LEGOAI/DATA
└───ecommerce_data
        olist_customers_dataset.csv
        olist_orders_dataset.csv
        olist_order_items_dataset.csv
        olist_order_payments_dataset.csv
        olist_order_reviews_dataset.csv
        olist_products_dataset.csv
        olist_sellers_dataset.csv
```
## Where to get it?
Binary installers for the latest released version are available at the [Python Package Index (PyPI)](https://pypi.org/project/)
```
# PyPI
> pip install legoai
```

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




