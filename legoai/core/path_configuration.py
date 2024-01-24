from dotenv import dotenv_values
import os

def prepare_path_configuration():
    
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
        "LOG_PATH":"logs",
        "LOG_LEVEL":"info"
    }


    loaded_path_config = dotenv_values(".env")
    file_abs_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    return {
        **default_path_config,
        **loaded_path_config,
        "MODEL_DEP_PATH":os.path.join(file_abs_path,"model","dependant"),
        "MODEL_METRICS_PATH":os.path.join(file_abs_path,"model","model_metrics"),
        "MODEL_OBJECTS_PATH":os.path.join(file_abs_path,"model","model_objects"),
        "MODEL_RESULTS_PATH":os.path.join(file_abs_path,"model","model_results"),
        "DATASET_PATH_URL":"http://localhost:8000/Lego_AI/inference/"
    }


PATH_CONFIG = prepare_path_configuration()
