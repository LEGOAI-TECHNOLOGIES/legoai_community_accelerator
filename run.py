from legoai import DataTypeIdentificationPipeline
from legoai.utils import download_default_dataset
from legoai.core.configuration import MODEL_CONFIG
import os
# this is an example of how to run a pretrained pipeline
if __name__ == "__main__":

    # Example to run inference

    # dataset_path = download_default_dataset()
    # di_pipeline = DataTypeIdentificationPipeline.pretrained_pipeline(openai_api_key=os.environ.get("OPENAI_KEY")

    # result = di_pipeline.predict(dataset_path=dataset_path,save_to="di_output.csv")
    # print(result[['column_name_clean','predicted_datatype_l1','predicted_datatype_l3']].head())

    # Example to train
    di = DataTypeIdentificationPipeline()
    di.train(model_version="40011")
