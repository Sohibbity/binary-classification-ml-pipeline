from datetime import datetime

import pandas as pd
import torch
from pandas import DataFrame

from Model.BinaryClassifierModel import BinaryClassifierModel


class ModelPredictor:
    def __init__(self, model: BinaryClassifierModel):
        self.model = model


    def run_inference(self, input_df: DataFrame):
        self.model.eval()

        # Y tensor is already dropped in preprocessing step
        # Remaining DF is only x tensors
        x_tensor = torch.FloatTensor(input_df.values)

        with torch.no_grad():
            outputs = self.model(x_tensor)  # generates raw, model output

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

        return results




