import io
import logging
import time
from datetime import datetime

import pandas as pd

from Config.Constants import S3_BUCKET_NAME, S3_DIR_FOR_INPUT_EVAL_DATA, S3_DIR_FOR_PREDICTIONS
from DataProcessing.PreProcessor import PreProcessor
from DataHandler.ProdDataHandler import ProdDataHandler
from Model import ModelPredictor
from Utils.Utils import log_stage

logger = logging.getLogger(__name__)
class ProdInferencePipeline:
    def __init__(
            self,
            prod_data_retriever: ProdDataHandler,
            model_predictor: ModelPredictor

    ):
        self.prod_data_retriever = prod_data_retriever
        self.model_predictor = model_predictor


    def prod_pipeline(self, s3_input_file_name: str, s3_predictions_output_file_name: str):
        # Creates Stream of input file for model eval

        s3_stream = self.prod_data_retriever.stream_inference_input_file(
        bucket=S3_BUCKET_NAME,
        key=f"{S3_DIR_FOR_INPUT_EVAL_DATA}/{s3_input_file_name}",
    )

        text_stream = io.TextIOWrapper(s3_stream, encoding='utf-8')
        chunked_reader = pd.read_csv(text_stream, chunksize=50, sep=";")

        # Chunks and creates preprocessed DF ready for model inference

        for chunk_id, chunk in enumerate(chunked_reader,start = 1):

            pre_processed_chunk = PreProcessor.preprocess_chunk(chunk)
            chunked_predictions = self.model_predictor.run_inference(input_df = pre_processed_chunk)

            # Write chunked predictions to s3
            self.prod_data_retriever.stream_write_file(
                bucket = S3_BUCKET_NAME,
                key = f"{S3_DIR_FOR_PREDICTIONS}/{s3_predictions_output_file_name}",
                df = chunked_predictions,
                chunk_id = chunk_id
            )
            log_stage("Successfully wrote predictions ", chunk_id)
