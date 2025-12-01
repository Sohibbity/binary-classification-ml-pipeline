from datetime import datetime

from Clients.ClientFactory import ClientFactory
from Config.Constants import Model_OUTPUT_PATH
from Config.LoggingConfig import configure_logging
from ETL.ProdDataHandler import ProdDataHandler
from Model.ModelPredictor import ModelPredictor
from Pipeline.ProdInferencePipeline import ProdInferencePipeline



def main_production():
    print("hello world")
    configure_logging()
    client_factory = ClientFactory()

    # Wrapper client to interact with S3
    prod_data_retriever = ProdDataHandler(client_factory)

    # Model for inference

    # ML Pipeline
    prod_inference_pipeline = ProdInferencePipeline(prod_data_retriever, client_factory.sagemaker_runtime_client)

    s3_input_file_name = 'bank_small.csv'
    s3_predictions_output_file_name = f"{s3_input_file_name}-{datetime.now().isoformat()}"

    prod_inference_pipeline.prod_pipeline(s3_input_file_name,s3_predictions_output_file_name)

if __name__ == "__main__":
    main_production()