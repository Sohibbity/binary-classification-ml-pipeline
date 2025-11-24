import pandas as pd

from Config.Constants import Model_OUTPUT_PATH, \
    EVAL_DATA_SOURCE_PATH, EVAL_DATA_OUTPUT_PATH
from DataHandler.LocalDataHandler import DataRetriever
from Pipeline.LocalInferencePipeline import InferencePipeline

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def main_local():
    print("hello world")
    # 'Data Ingestion'
    DataRetriever.load_data(
        source_file_path = EVAL_DATA_SOURCE_PATH,
        output_file_path = EVAL_DATA_OUTPUT_PATH)

    # Inference Pipeline
    inference_pipeline = InferencePipeline(
        model_path = Model_OUTPUT_PATH,
        input_data = EVAL_DATA_OUTPUT_PATH
    )

    # Evaluates input data set and outputs results under the /EvaluatedData Dir
    # Output file name includes the name of the file evaluated along with timestamp
    inference_pipeline.pipeline()

if __name__ == "__main__":
    main_local()