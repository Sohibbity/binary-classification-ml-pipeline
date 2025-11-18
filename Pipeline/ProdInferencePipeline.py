import io
import logging
import time
from datetime import datetime

import pandas as pd

from Config.Constants import S3_BUCKET_NAME, S3_DIR_FOR_INPUT_EVAL_DATA, S3_DIR_FOR_PREDICTIONS
from DataProcessing.PreProcessor import PreProcessor
from DataHandler.ProdDataHandler import ProdDataHandler
from Model import ModelPredictor
from Utils.Utils import log_stage, log_retry, log_chunk_failure

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
        failed_chunks = []
        successful_chunks = []
        for chunk_id, chunk in enumerate(chunked_reader, start=1):
            """
            This is scrappy (AND TEMPORARY!), and is simply to bridge the gap 
            between an ideal system in production and this prototype
            Under ideal circumstances these are air flow tasks spun up on a worker pool
            failed batches are retried on a separate pool 
            and SLA is calculated periodically, 
            if too many tasks fail then independent task retry is useless 
            In that case the entire pipeline must be restarted 
            TODO: Explore airflow dag setup, as singular task = chunk of operations or 1 task = 1 specific operation
            i.e 1 task for preprocessing, 1 task for inference, 1 task for writing to s3, or 1 task  for all 3
            """
            early_abort_threshold = 0.9 # sla is .98, tuned slightly below to prevent pre-emptively aborting workflow
            check_interval = 5
            try:
                pre_processed_chunk = PreProcessor.preprocess_chunk(chunk)
            except Exception as e:
                log_chunk_failure('Preprocessing DF', chunk_id,
                                  "Dropping curr chunk",
                                  f"{e}")
                failed_chunks.append(chunk_id)
                continue

            chunk_succeeded = False

            for r in range(1,4):
                try:
                    if chunk_id % 10 == 0:
                        raise ValueError('simulating failed chunk')
                    chunked_predictions = self.model_predictor.run_inference(input_df=pre_processed_chunk)

                    self.prod_data_retriever.stream_write_file(
                        bucket=S3_BUCKET_NAME,
                        key=f"{S3_DIR_FOR_PREDICTIONS}/{s3_predictions_output_file_name}",
                        df=chunked_predictions,
                        chunk_id=chunk_id
                    )
                    chunk_succeeded = True
                    break
                except Exception as e:
                    log_retry('Predictions failed to write to s3', chunk_id, r, max_attempts = 3)

            if chunk_succeeded:
                log_stage(f"Writing Predictions to S3", chunk_id)
                successful_chunks.append(chunk_id)
            else:
                failed_chunks.append(chunk_id)
                # allow for first 20 tasks to processs , avoiding cold start issues
            if chunk_id > 20 and chunk_id % check_interval == 0:
                current_success_rate = len(successful_chunks) / chunk_id
                if current_success_rate < early_abort_threshold:
                    raise ValueError('ABANDOING WORKFLOW, SLA NOT MET, PLEASE RE RUN ENTIRE WORKFLOW FROM START')

        total_chunks = len(successful_chunks) + len(failed_chunks)
        print(f"total chunks: {total_chunks}")
        print(f"failed chunks: {failed_chunks}")
        print(f"successful chunks: {successful_chunks}")
        success_rate = len(successful_chunks) / total_chunks
        if success_rate < 0.98:
            raise Exception(f"Pipeline failed: Success rate {success_rate:.1%} below 98% SLA threshold")


