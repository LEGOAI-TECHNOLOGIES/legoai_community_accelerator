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
