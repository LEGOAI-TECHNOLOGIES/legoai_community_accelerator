import yaml
# from legoai.core.path_configuration import PATH_CONFIG
import os
import sys

def load_default_model_configuration(file_path:str = 'legoai',file_name:str = "config.yaml"):
    model_yaml_path = os.path.join(file_path,file_name)
    config = {}
    try:       
        with open(model_yaml_path,"r") as default_config_file:
            config = yaml.safe_load(default_config_file)
        return config
    
    except FileNotFoundError:
        print(f"\n[!] default configuration file not found at {model_yaml_path}")
        sys.exit(-1)


MODEL_CONFIG = load_default_model_configuration()

def load_model_configuration(file_path:str,file_name:str):

    model_yaml_path = os.path.join(file_path,file_name)
    try:
        if not file_name.endswith(".yaml"):
            raise Exception("[!] configuration file must be yaml")
        else:
            with open(model_yaml_path,"r") as custom_config_file:
                MODEL_CONFIG = yaml.safe_load(custom_config_file)
                print("[*] loaded custom model configuration")

    except Exception as ex:
        print(ex.__str__() + "... loading default configuration")

        MODEL_CONFIG = load_default_model_configuration()


