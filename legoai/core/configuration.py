import yaml
from dotenv import dotenv_values

import os
import sys


def load_model_configuration():
    """
    - loads the configuration needed for the modules from config.yaml or default configuration

    Parameters
    ----------
        
    Returns
    -------
    dictionary of configuration either loaded from default path or through used defined config.yaml file
    """

    dir_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    config_file = "config.yaml"
    default_path = os.path.join(dir_path, config_file)

    default_config = {}
    custom_config = {}
    try:
        with open(default_path, "r") as default_config_file:
            default_config = yaml.safe_load(default_config_file)

        # check if user has its own config file
        if os.path.exists(config_file):
            with open(config_file, "r") as custom_config_file:
                custom_config = yaml.safe_load(custom_config_file)

    except FileNotFoundError:
        print(f"\n[!] configuration file not found")
        sys.exit(-1)


    return {
        **default_config,
        **custom_config
    }


MODEL_CONFIG = load_model_configuration()

L1_FINAL_FOLDER_NAME = "datatype_l1_identification"


def check_path_exists_or_not(path_config: dict):
    """
    - Checks if all the path exists or not from either .env files or default path config obtained from load_path_configuration() if not creates them
    Parameters
    ----------
    path_config (dict): dictionary of all the path to relevant folders

    Returns
    -------

    """

    all_requirement_paths = path_config.keys()
    main_entry_path = path_config["CONTAINER_PATH"]

    non_required = ["CONTAINER_PATH", "DATASET_PATH_URL"]

    # checks if necessary path is made or not if not then it creates one
    if not os.path.exists(main_entry_path):
        os.makedirs(main_entry_path)

    # check all the other paths
    for inner_path in set(all_requirement_paths):
        inner_path = os.path.normpath(inner_path)
        if inner_path not in non_required:
            special_case = ""
            if not inner_path.startswith("INF"):
                special_case += "datatype_l1_identification"

            full_inner_path = os.path.normpath(os.path.join(main_entry_path, path_config[inner_path], special_case))
            if not os.path.exists(full_inner_path):
                # not exists then create the paths
                os.makedirs(full_inner_path)

def load_path_configuration(**kwargs):
    """
    - Loads the necessary path configuration for the program from .env file or either loads the default value

    Returns
    -------
    final dictionary of all relevant paths
    """
    default_path_config = {
        "CONTAINER_PATH":"datatype_identification_training",
        "INT_DATA_PATH": "intermediate",
        "ANALYTICAL_DATA_PATH": "analytical_data",
        # "MODEL_DEP_PATH": os.path.join(*"model", "dependant"),
        "MODEL_METRICS_PATH": os.path.join(*"model", "model_metrics"),
        "MODEL_OBJECTS_PATH": os.path.join(*"model", "model_objects"),
        "MODEL_RESULTS_PATH": os.path.join(*"model", "model_results"),
    }

    custom_path_config = dotenv_values(".env")
    # checks all requirement directories , if not then it creates one
    # check_path_exists_or_not({**default_path_config, **custom_path_config})

    # file_abs_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

    return {
        **default_path_config,
        **custom_path_config
    }


PATH_CONFIG = load_path_configuration()
