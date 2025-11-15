import pandas as pd

from Config.Constants import DATA_SOURCE_PATH, OUTPUT_FILE_PATH, DF_ROWS, Model_OUTPUT_PATH
from Data.DataRetriever import DataRetriever
from DataProcessing.PreProcessor import PreProcessor
from ModelBuilding.BinaryClassifierModel import BinaryClassifierModel
from ModelBuilding.ModelRetriever import ModelRetriever
from ModelBuilding.ModelTrainer import ModelTrainer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def main():
    print("hello world")
    # 'Data Ingestion'
    DataRetriever.load_data(DATA_SOURCE_PATH, OUTPUT_FILE_PATH)

    # pre-process data set
    processed_df = PreProcessor.preprocess_csv(OUTPUT_FILE_PATH)
    print(f"{processed_df.head(5)}")

    print("Begin Model Training")

    model_trainer = ModelTrainer(model=BinaryClassifierModel(input_size= DF_ROWS))

    # Train and save Model
    model_trainer.train_model(processed_df = processed_df)

    # Load Model from local file source, TBD actual retrieval from Blob Storage Service
    loaded_model = ModelRetriever().load_model(stored_model_file_path = Model_OUTPUT_PATH)
    print(loaded_model)

if __name__ == "__main__":
    main()