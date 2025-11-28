from pathlib import Path
DATA_DIR = Path("/Users/soheeb/PycharmProjects/BinaryClassifier/Data")
INPUT_DATA_DIR = DATA_DIR / "InputData"
EVALUATED_DATA_DIR = DATA_DIR / "EvaluatedData"

# Data Ingestion
TRAINING_DATA_SOURCE_PATH = Path("/Users/soheeb/Desktop/uci_bank_marketing/bank/bank-full.csv")
TRAINING_DATA_OUTPUT_FILE_PATH = "bank-full.csv"   # <-- relative, good

# Model Eval
EVAL_DATA_SOURCE_PATH = Path("/Users/soheeb/Desktop/uci_bank_marketing/bank/bank.csv")
EVAL_DATA_OUTPUT_PATH = "bank.csv"  # <-- relative

# Evaluated Data Output Paths
test = '/Users/soheeb/PycharmProjects/BinaryClassifier/Data/EvaluatedData'


# AWS

S3_BUCKET_NAME = 'ml-inference-data-soheeb'
S3_DIR_FOR_INPUT_EVAL_DATA = 'input_eval_data'
S3_DIR_FOR_PREDICTIONS = 'output_predictions'



# General Constants
Y_AXIS = 'y'
X_AXIS = 'x'

# Model Training
DF_ROWS = 8
Model_OUTPUT_PATH = '/Users/soheeb/PycharmProjects/BinaryClassifier/Models/model-v1.pth'

# Model names:

# Initial 2-MLP Model used for binary classification of predicting subscriber sign ups
MODEL_V1 = 'model-v1'
