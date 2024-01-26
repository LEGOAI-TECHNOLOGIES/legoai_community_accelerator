import yaml
import os
import sys

def load_model_configuration():
    '''
        loads the configuration needed for the modules

        Parameters:
            None
        
        Returns:
            dictionary of configuration either loaded from default path or through used defined config.yaml file
    '''
   
    dir_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    config_file = "config.yaml"
    default_path = os.path.join(dir_path,config_file)
    

    default_config = {}
    custom_config = {}
    try:       
        with open(default_path,"r") as default_config_file:
            default_config = yaml.safe_load(default_config_file)

        # check if user has its own config file
        if os.path.exists(config_file):
            with open(config_file,"r") as custom_config_file:
                custom_config = yaml.safe_load(custom_config_file)
                
    except FileNotFoundError:
        print(f"\n[!] configuration file not found")
        sys.exit(-1)

    return {
        **default_config,
        **custom_config
        }

MODEL_CONFIG = load_model_configuration()

from dotenv import dotenv_values

import os

def load_path_configuration():
    
    default_path_config = {
        "CONTAINER_PATH" : os.path.join("data", "Lego_AI"),
        "GT_PATH" : os.path.join(*"input/ground_truth".split("/")),
        "RAW_DATA_PATH":os.path.join(*"input/raw_data".split("/")),
        "DEV_REF_DATA_PATH":os.path.join(*"input","reference_data"),
        "INT_DATA_PATH":"intermediate",
        "ANALYT_DATA_PATH": "analytical_data",
        "INF_DATA_PATH":os.path.join("inference","input"),
        "INF_OUT_PATH":os.path.join("inference","test_output"),
        "INF_REF_DATA_PATH":os.path.join(*"inference","reference_files"),
        "MODEL_DEP_PATH":os.path.join("model","dependant"),
        "MODEL_METRICS_PATH":os.path.join("model","model_metrics"),
        "MODEL_OBJECTS_PATH":os.path.join("model","model_objects"),
        "MODEL_RESULTS_PATH":os.path.join("model","model_results"),
        "LOG_PATH":"logs",
        "LOG_LEVEL":"info",
        "DATASET_PATH_URL":"http://localhost:8000/Lego_AI/inference/"
    }


    loaded_path_config = dotenv_values(".env")
    # file_abs_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    return {
        **default_path_config,
        **loaded_path_config,
      
        
    }


PATH_CONFIG = load_path_configuration()