from legoai import DataTypeIdentificationPipeline
from legoai.core.configuration import MODEL_CONFIG

if __name__ == "__main__":
    di_pipeline = DataTypeIdentificationPipeline.pretrained_pipeline(open_api_key=MODEL_CONFIG["L3PARAMS"]["OPENAI_API_KEY"])
    di_pipeline.
    result = di_pipeline(dataset_name="ecommerce_data",save_to="di_output.csv")
    print(result[['column_name_clean','predicted_datatype_l1','predicted_datatype_l3']].head())