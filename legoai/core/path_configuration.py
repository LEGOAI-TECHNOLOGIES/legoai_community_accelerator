from dotenv import dotenv_values
import os

def prepare_path_configuration():
    file_abs_path = os.path.abspath(os.path.dirname(__file__))
    default_path_config = {
        "CONTAINER_PATH" : file_abs_path.removesuffix("\core")+"\Lego_AI",
        "GT_PATH" : "data/input/ground_truth",
        "RAW_DATA_PATH":"data/input/raw_data",
        "DEV_REF_DATA_PATH":"data/input/reference_data",
        "INT_DATA_PATH":"data/intermediate",
        "ANALYT_DATA_PATH": "data/analytical_data",
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
        **loaded_path_config,
        **default_path_config
    }


PATH_CONFIG = prepare_path_configuration()
