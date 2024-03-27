<p align="center">
      <img src="documentation/gray%20LEG0AI%20Cymk%20color%20JPG.jpg" alt="LegoAI Logo" width="350">
 <h3 align="center"><i>Empowering Business Users With Self Serve Analytics</i></h3> 

## What is it ?
This is a cutting-edge project leveraging advanced Machine Learning technologies to accurately discern and classify data types from various values. Designed to enhance data preprocessing and analysis pipelines, this tool automates the often tedious and error-prone task of manually identifying data types.

## Table of contents
- [Main Features](#main-features)
- [Where to get it](#where-to-get-it)
- [Getting Started](#getting-started)
- [Performace](#performace)
- [License](#license)
- [Documentation](#documentation)
- [Contributing](#contributing)
## Main Features
The project has different pipelines ... currently only released module is:
- **Data Type Identification**
    > - Classifying datatypes from various values easily through trained ML model.
    > - Has two models, L1 model (_uses Classifier_) that identifies normal datatypes  ( integer, float, alphanumeric, range_type, date & time, open_ended_text, short_ended_text)
    > - L2 model (_uses LLM_) further classifies L1 datatype result that are integer or float to measure or dimension and date & time into one of many date-time formats like (YYYY:mm:dd H:m:s).
- coming soon ...  

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
> **openai_api_key** is required if you want to use L3 model else leave the field then inference will only run for L1 model


## Performance
### Data Type Identification
<table>
<tr>
  <th>Dataset</th> 
  <th colspan="2"></th> 
  <th> Precision</th> 
  <th>Recall</th>
  <th> F1 Score </th> 
  <th> Support </th>
</tr>
<tr> 
  <th rowspan="10"> 
     <a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce">Ecommerce</a> 
</th>
</tr>
<tr>
  <th rowspan="6"> Class</th>
 <td> alphanumeric </td>
 <td> 1.0 </td>
 <td> 1.0 </td>
 <td> 1.0 </td>
 <td> 12.0 </td>
</tr>
<tr>
 <td> close_ended_text </td>
 <td> 0.88 </td>
 <td> 1.0 </td>
 <td> 0.93 </td>
 <td> 7.0 </td>
</tr>
<tr>
 <td> date & time </td>
 <td> 1.0 </td>
 <td> 1.0 </td>
 <td> 1.0 </td>
 <td> 8.0 </td>
</tr>
<tr>
 <td> float </td>
 <td> 1.0 </td>
 <td> 0.75 </td>
 <td> 0.86 </td>
 <td> 4.0 </td>
</tr>
<tr>
 <td> integer </td>
 <td> 0.92 </td>
 <td> 1.0 </td>
 <td> 0.96 </td>
 <td> 12.0 </td>
</tr>
<tr>
 <td> open_ended_text </td>
 <td> 1.0 </td>
 <td> 0.75 </td>
 <td> 0.86 </td>
 <td> 4.0 </td>
</tr>
<tr>
  <td rowspan="3"></td>
 <td> accuracy </td>
 <td> 0.96 </td>
 <td> 0.96 </td>
 <td> 0.96 </td>
 <td> 0.96 </td>
</tr>
<tr>
 <td> macro avg </td>
 <td> 0.97 </td>
 <td> 0.92 </td>
 <td> 0.93 </td>
 <td> 47.0 </td>
</tr>
<tr>
 <td> weighted avg </td>
 <td> 0.96 </td>
 <td> 0.96 </td>
 <td> 0.96 </td>
 <td> 47.0 </td>
</tr>
<tr> 
<th rowspan="11">
 <a href="https://mitre.box.com/shared/static/aw9po06ypfb9hrau4jamtvtz0e5ziucz.zip">
   Health Care
  </a>
</th> 
</tr>
<tr>
 <th rowspan="7"> Class </th>
 <td> alphanumeric </td>
 <td> 0.58 </td>
 <td> 0.92 </td>
 <td> 0.71 </td>
 <td> 61.0 </td>
</tr>
<tr>
 <td> close_ended_text </td>
 <td> 0.69 </td>
 <td> 0.88 </td>
 <td> 0.77 </td>
 <td> 40.0 </td>
</tr>
<tr>
 <td> date & time </td>
 <td> 0.75 </td>
 <td> 0.21 </td>
 <td> 0.32 </td>
 <td> 29.0 </td>
</tr>
<tr>
 <td> float </td>
 <td> 0.79 </td>
 <td> 0.71 </td>
 <td> 0.75 </td>
 <td> 31.0 </td>
</tr>
<tr>
 <td> integer </td>
 <td> 0.84 </td>
 <td> 0.85 </td>
 <td> 0.84 </td>
 <td> 54.0 </td>
</tr>
<tr>
 <td> open_ended_text </td>
 <td> 0.6 </td>
 <td> 0.1 </td>
 <td> 0.17 </td>
 <td> 30.0 </td>
</tr>
<tr>
 <td> others </td>
 <td> 0.17 </td>
 <td> 0.25 </td>
 <td> 0.2 </td>
 <td> 4.0 </td>
</tr>
<tr>
 <td rowspan="3"> </td>
 <td> accuracy </td>
 <td> 0.68 </td>
 <td> 0.68 </td>
 <td> 0.68 </td>
 <td> 0.68 </td>
</tr>
<tr>
 <td> macro avg </td>
 <td> 0.63 </td>
 <td> 0.56 </td>
 <td> 0.54 </td>
 <td> 249.0 </td>
</tr>
<tr>
 <td> weighted avg </td>
 <td> 0.69 </td>
 <td> 0.68 </td>
 <td> 0.64 </td>
 <td> 249.0 </td>
</tr>
<tr> <th rowspan="10"> 
    <a href="https://www.kaggle.com/datasets/dillonmyrick/bike-store-sample-database"> Bike Retail </a> 
</th> 
</tr>
<tr>
 <th rowspan="6"> Class </th>
 <td> alphanumeric </td>
 <td> 1.0 </td>
 <td> 1.0 </td>
 <td> 1.0 </td>
 <td> 5.0 </td>
</tr>
<tr>
 <td> close_ended_text </td>
 <td> 1.0 </td>
 <td> 0.5 </td>
 <td> 0.67 </td>
 <td> 4.0 </td>
</tr>
<tr>
 <td> date & time </td>
 <td> 0.75 </td>
 <td> 1.0 </td>
 <td> 0.86 </td>
 <td> 3.0 </td>
</tr>
<tr>
 <td> float </td>
 <td> 1.0 </td>
 <td> 1.0 </td>
 <td> 1.0 </td>
 <td> 3.0 </td>
</tr>
<tr>
 <td> integer </td>
 <td> 1.0 </td>
 <td> 0.96 </td>
 <td> 0.98 </td>
 <td> 26.0 </td>
</tr>
<tr>
 <td> open_ended_text </td>
 <td> 0.85 </td>
 <td> 1.0 </td>
 <td> 0.92 </td>
 <td> 11.0 </td>
</tr>
<tr>
 <td rowspan="3"> </td>
 <td> accuracy </td>
 <td> 0.94 </td>
 <td> 0.94 </td>
 <td> 0.94 </td>
 <td> 0.94 </td>
</tr>
<tr>
 <td> macro avg </td>
 <td> 0.93 </td>
 <td> 0.91 </td>
 <td> 0.9 </td>
 <td> 52.0 </td>
</tr>
<tr>
 <td> weighted avg </td>
 <td> 0.95 </td>
 <td> 0.94 </td>
 <td> 0.94 </td>
 <td> 52.0 </td>
</tr>
</table>
<p align="center">
  <img src="documentation/Execution_Char.png" alt="DI Execution Chart" width="350">
</p>


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




