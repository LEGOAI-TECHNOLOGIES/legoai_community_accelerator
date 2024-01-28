from legoai import DataTypeIdentificationPipeline
from legoai.utils import download_default_dataset
from legoai.core.configuration import MODEL_CONFIG
# this is an example of how to run a pretrained pipeline
if __name__ == "__main__":
    dataset_name = download_default_dataset()
    # print(dataset_name)
    di_pipeline = DataTypeIdentificationPipeline.pretrained_pipeline(openai_api_key=MODEL_CONFIG["L3PARAMS"]["OPENAI_API_KEY"])
    result = di_pipeline(dataset_name=dataset_name,save_to="di_output.csv")
    print(result[['column_name_clean','predicted_datatype_l1','predicted_datatype_l3']].head())