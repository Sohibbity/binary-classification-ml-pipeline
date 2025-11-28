import io
import json
import logging
from datetime import datetime

import pandas as pd

from Config.Constants import S3_BUCKET_NAME, S3_DIR_FOR_INPUT_EVAL_DATA, S3_DIR_FOR_PREDICTIONS
from DataProcessing.PreProcessor import PreProcessor
from DataHandler.ProdDataHandler import ProdDataHandler
from Model import ModelPredictor
from Utils.Utils import log_stage, log_retry, log_chunk_failure
from sagemaker_deploy.DeploySagemakerEndpoint import ENDPOINT_NAME

logger = logging.getLogger(__name__)


class ProdInferencePipeline:
    """
    Production inference pipeline prototype demonstrating S3 streaming and batch processing.

    Note: This is a monolithic implementation that mixes orchestration and business logic.
    In a mature production system, these steps will be decomposed into separate Airflow tasks:
        - Task 1: Stream & preprocess chunks
        - Task 2: Run model inference
        - Task 3: Write predictions to S3

    Each task would run on isolated worker pools with independent retry mechanisms and
    monitoring. This prototype establishes the core pipeline logic before adding orchestration.

    Completed: S3 Streaming
    Pending: Model Inference via dedicated Sagemaker Inference Endpoints
    TODO: Set up Postgres for storage and versioning of model outputs
    TODO: Migrate to Airflow for task orchestration
    TODO: Implement model performance monitoring and drift detection
    TODO: Add alerting for SLA breaches and pipeline failures
    """
    def __init__(
            self,
            prod_data_retriever: ProdDataHandler,
            model_predictor: ModelPredictor,
            sagemaker_runtime_client

    ):
        self.prod_data_retriever = prod_data_retriever
        self.model_predictor = model_predictor
        self.sagemaker_runtime_client = sagemaker_runtime_client

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
            # Prime example of why we don't mix orchestration logic with business logic
            # TBD Airflow
            early_abort_threshold = 0.0 # sla is .98, tuned slightly below to prevent pre-emptively aborting workflow
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

                    formatted_for_inference = pre_processed_chunk.values.tolist()
                    payload = json.dumps(formatted_for_inference)
                    print(f'type: {type(payload)}')
                    print('Begin inference chunks')
                    print(f'122201, payload: {payload}')
                    print('---------------------------------------------------------')
                    # chunked_predictions = self.model_predictor.run_inference(input_df=pre_processed_chunk)
                    # print(f'1001.1 chunked_pred {chunked_predictions}')

                    print(f"Calling endpoint: {ENDPOINT_NAME}")
                    print(f"Payload: {payload}")

                    # Invoke endpoint
                    inference_predictions = self.sagemaker_runtime_client.invoke_endpoint(
                        EndpointName=ENDPOINT_NAME,
                        ContentType='application/json',
                        Accept='application/json',
                        Body=payload
                    )

                    result = json.loads(inference_predictions['Body'].read().decode())

                    # Create DataFrame (matching your local format)
                    chunked_predictions = pd.DataFrame({
                        'prediction': result['predictions'],
                        'confidences': result['confidences'],
                        'timestamp': datetime.now().isoformat(),
                        'model_version': 'v1.0.0.0'
                    })
                    chunked_predictions.index.name = 'input_row_id'


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


