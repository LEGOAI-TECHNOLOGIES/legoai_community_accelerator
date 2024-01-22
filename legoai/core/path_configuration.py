from dotenv import dotenv_values
import os

def prepare_path_configuration():
    file_abs_path = os.path.abspath(os.path.dirname(__file__))
    default_path_config = {
        "CONTAINER_PATH" : os.path.join("data", "Lego_AI"),
        "GT_PATH" : os.path.join(*"input/ground_truth".split("/")),
        "RAW_DATA_PATH":os.path.join(*"input/raw_data".split("/")),
        "DEV_REF_DATA_PATH":"input/reference_data",
        "INT_DATA_PATH":"intermediate",
        "ANALYT_DATA_PATH": "analytical_data",
        "MODEL_DEP_PATH":"model/dependant",
        "MODEL_METRICS_PATH":"model/model_metrics",
        "MODEL_OBJECTS_PATH":"model/model_objects",
        "MODEL_RESULTS_PATH":"model/model_results",
        "INF_DATA_PATH":"inference/input",
        "INF_OUT_PATH":"inference/test_output",
        "INF_REF_DATA_PATH":"inference/reference_files",
        "LOG_PATH":"logs",
        "LOG_LEVEL":"info"
    }


    loaded_path_config = dotenv_values(".env")
    
    return {
        **default_path_config,
        **loaded_path_config

    }


PATH_CONFIG = prepare_path_configuration()
