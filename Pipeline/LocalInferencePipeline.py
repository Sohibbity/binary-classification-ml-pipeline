from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from Config.Constants import EVAL_DATA_OUTPUT_PATH, \
    EVALUATED_DATA_DIR, INPUT_DATA_DIR
from DataProcessing.PreProcessor import PreProcessor
from Model.ModelRetriever import ModelRetriever


class InferencePipeline:
    def __init__(self, model_path: Path, input_data: Path):
        self.model_path = model_path
        self.input_data = input_data


    def pipeline(self):
        # load model
        model = ModelRetriever().load_model(stored_model_file_path = self.model_path)
        model.eval()

        # preprocess data
        preprocessed_df = PreProcessor.preprocess_csv(Path(INPUT_DATA_DIR / self.input_data))
        # Sample data contains prepopulated y, hence drop target column

        x = preprocessed_df.drop('y', axis=1)
        x_tensor = torch.FloatTensor(x.values)

        #  CAll Model on dataset
        with torch.no_grad():
            outputs = model(x_tensor) # generates raw, model output
            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)  # Convert to probabilities
            predictions = torch.argmax(probabilities, dim=1)  # Get predicted class (0 or 1)
            confidences = torch.max(probabilities, dim=1).values  # Get confidence score

        results = pd.DataFrame({
            'prediction': predictions.numpy(),
            'confidences': confidences,
            'timestamp': datetime.now().isoformat(),
            'model_version': 'v1.0.0.0'
        })

        results.index.name = 'input_row_id'

        timestamp = datetime.now().isoformat().replace(":", "-")

        base_name = Path(EVAL_DATA_OUTPUT_PATH).stem  # "bank"
        extension = Path(EVAL_DATA_OUTPUT_PATH).suffix  # ".csv"

        final_filename = f"{base_name}_{timestamp}{extension}"

        output_csv_path = Path(EVALUATED_DATA_DIR) / final_filename
        results.to_csv(output_csv_path, index=True)
        print(f"Predictions saved to {output_csv_path}")



