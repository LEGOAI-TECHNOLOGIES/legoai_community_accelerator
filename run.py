from legoai.pipeline import DataTypeIdentificationPipeline
from legoai.modules.datatype_identification import MODEL_CONFIG


if __name__ == "__main__":
    di_pipeline = DataTypeIdentificationPipeline(
        dataset_name="ecommerce_data",
        openai_api_key= MODEL_CONFIG["L3PARAMS"]["OPENAI_API_KEY"] 
        ) 
   
    di_pipeline.run_pretrained_pipeline(save_to="output.csv")