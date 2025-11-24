from datetime import datetime

from Clients.ClientFactory import ClientFactory
from Config.Constants import Model_OUTPUT_PATH
from Config.LoggingConfig import configure_logging
from DataHandler.ProdDataHandler import ProdDataHandler
from Model.ModelPredictor import ModelPredictor
from Model.ModelRetriever import ModelRetriever
from Pipeline.ProdInferencePipeline import ProdInferencePipeline



def main_production():
    print("hello world")
    configure_logging()
    client_factory = ClientFactory()

    # Wrapper client to interact with S3
    prod_data_retriever = ProdDataHandler(client_factory)

    # Model for inference
    model_predictor = ModelPredictor(ModelRetriever().load_model(stored_model_file_path=Model_OUTPUT_PATH))

    # ML Pipeline
    prod_inference_pipeline = ProdInferencePipeline(prod_data_retriever,model_predictor)

    s3_input_file_name = 'bank.csv'
    s3_predictions_output_file_name = f"{s3_input_file_name}-{datetime.now().isoformat()}"

    prod_inference_pipeline.prod_pipeline(s3_input_file_name,s3_predictions_output_file_name)


if __name__ == "__main__":
    main_production()