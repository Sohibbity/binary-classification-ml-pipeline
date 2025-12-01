from datetime import datetime
from pathlib import Path

import torch

from Config.Constants import EVAL_DATA_OUTPUT_PATH, \
    EVALUATED_DATA_DIR, INPUT_DATA_DIR
from ETL.PreProcessor import PreProcessor
from Model.ModelPredictor import ModelPredictor


class LocalInferencePipeline:
    """
    Local inference pipeline for model evaluation and testing.
    Runs a localized version of the prod pipeline
    Use this for debugging/testing/model eval
    """
    def __init__(self, input_data: Path, model_predictor: ModelPredictor):
        self.input_data = input_data
        self.model_predictor = model_predictor

    def pipeline(self):

        # Preprocess data:
        # Drop Y column (prediction)
        # Convert remaining df to tensors
        preprocessed_df = ((PreProcessor.preprocess_csv(Path(INPUT_DATA_DIR / self.input_data)))
                           .drop('y', axis=1))

        x_tensor = torch.FloatTensor(preprocessed_df.values)

        results = self.model_predictor.run_local_inference(x_tensor)

        results.index.name = 'input_row_id'

        timestamp = datetime.now().isoformat().replace(":", "-")

        base_name = Path(EVAL_DATA_OUTPUT_PATH).stem  # "bank"
        extension = Path(EVAL_DATA_OUTPUT_PATH).suffix  # ".csv"

        final_filename = f"{base_name}_{timestamp}{extension}"

        output_csv_path = Path(EVALUATED_DATA_DIR) / final_filename
        results.to_csv(output_csv_path, index=True)
        print(f"Predictions saved to {output_csv_path}")
