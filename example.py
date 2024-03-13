from legoai import DataTypeIdentificationPipeline
from legoai.utils import download_default_dataset
from legoai.core.configuration import MODEL_CONFIG
import os
# this is an example of how to run a pretrained pipeline
if __name__ == "__main__":

    # Example to run inference

    # dataset_path = "D:\Lego AI\DI_OPENSOURCE\default_data_29022024\inference\inference_repo\ecommerce_data"

    # load pretrained version of the pipeline
    # di_pipeline = DataTypeIdentificationPipeline.pretrained_pipeline(
    #     encoder_path="D:\Lego AI\DI_OPENSOURCE\legoai\model\model_objects\datatype_l1_identification\di_l1_classifier_encoder_13052023.pkl",
    #     model_path = "D:\Lego AI\DI_OPENSOURCE\legoai\model\model_objects\datatype_l1_identification\di_l1_classifier_xgb_13052023.pkl"
    # )

    # run the prediction
    # result = di_pipeline.predict(input_path=dataset_path,output_path="D:/di_opensource")
    # display result
    # print(result.head())


    # Example to train

    di = DataTypeIdentificationPipeline()
    # provide data path for training and its corresponding ground truth or labels
    dataset_path = r"D:\Lego AI\DI_OPENSOURCE\training_directory\Lego_AI\input\raw_data"
    ground_truth_path = r"D:\Lego AI\DI_OPENSOURCE\training_directory\Lego_AI\input\ground_truth"

    # final output path to save intermediate files, classification and confusion matrix reports , and encoder and classifier model.
    output_path = r"D:\datatype_identification_training"
    #give model version to save the final encoders and classifier model under the given version
    model_version = "13052023"
    di.train(input_path=dataset_path,gt_path=ground_truth_path,
             output_path= output_path,
             model_version=model_version)
    