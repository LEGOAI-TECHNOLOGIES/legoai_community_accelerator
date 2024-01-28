import yaml
from dotenv import dotenv_values

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


def check_path_exists_or_not(path_config:dict):
    
    all_requirement_paths = path_config.keys()
    main_entry_path = path_config["CONTAINER_PATH"]
    non_required = ["CONTAINER_PATH","DATASET_PATH_URL"]
    # checks if necessary path is made or not if not the it creates one
    if not os.path.isdir(main_entry_path):
        os.makedirs(main_entry_path)

    # check all the other paths
    for inner_path in set(all_requirement_paths):
        
        if inner_path not in non_required:
            full_inner_path = os.path.join(main_entry_path,path_config[inner_path])
            if not os.path.isdir(full_inner_path):
                # not exists then create the paths
                os.makedirs(full_inner_path)
        





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
        "DATASET_PATH_URL":"http://localhost:8000/Lego_AI/inference/"
    }


    loaded_path_config = dotenv_values(".env")
    # checks all requirement directories , if not then it creates one
    check_path_exists_or_not({**default_path_config,**loaded_path_config})

   
    # file_abs_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    return {
        **default_path_config,
        **loaded_path_config,
        }


PATH_CONFIG = load_path_configuration()