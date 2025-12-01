import pandas as pd

from Config.Constants import Model_OUTPUT_PATH, \
    EVAL_DATA_SOURCE_PATH, EVAL_DATA_OUTPUT_PATH
from ETL.LocalDataHandler import LocalDataHandler
from Model.ModelPredictor import ModelPredictor
from Pipeline.LocalInferencePipeline import LocalInferencePipeline

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

"""
Local Inference Runner

This module provides a local debugging entry point for models that are deployed
to production. It allows engineers to replicate the inference pipeline locally
by:

1. Downloading the model's `.pth` artifact from production (S3 or the model
   registry).
2. Running the same preprocessing and inference logic used in the deployed
   SageMaker endpoint.
3. Producing an evaluated output file under `/EvaluatedData` with predictions,
   confidences, timestamps, and metadata.

This script is intended for:
- Investigating production inference issues
- Validating new model artifacts before deployment
- Comparing local vs. remote inference behavior
- Rapid iteration on preprocessing or postprocessing logic

To use:
1. Place the `.pth` model file in the configured `Model_OUTPUT_PATH`.
2. Place or download the evaluation CSV into `EVAL_DATA_SOURCE_PATH`.
3. Run `python main_local.py` to execute the full local inference pipeline.
"""

def main_local():
    print("hello world")
    # 'Data Ingestion'
    LocalDataHandler.load_data(
        source_file_path = EVAL_DATA_SOURCE_PATH,
        output_file_path = EVAL_DATA_OUTPUT_PATH)

    # Local Inference Pipeline
    local_inference_pipeline = LocalInferencePipeline(
        input_data=EVAL_DATA_OUTPUT_PATH,
        model_predictor=(ModelPredictor.from_path(Model_OUTPUT_PATH))
    )

    # Evaluates input data set and outputs results under the /EvaluatedData Dir
    # Output file name includes the name of the file evaluated along with timestamp
    local_inference_pipeline.pipeline()

if __name__ == "__main__":
    main_local()