from legoai.pipeline import DataTypeIdentificationPipeline
from legoai.modules.datatype_identification import MODEL_CONFIG

if __name__ == "__main__":
    di_pipeline = DataTypeIdentificationPipeline.pretrained_pipeline(open_api_key=MODEL_CONFIG["L3PARAMS"]["OPENAI_API_KEY"])
    result = di_pipeline(dataset_name="ecommerce_data")
    print(result[['column_name_clean','predicted_datatype_l1','predicted_datatype_l3']])