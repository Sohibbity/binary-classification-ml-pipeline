from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from pandas import DataFrame
from torch import FloatTensor

from Model.BinaryClassifierModel import BinaryClassifierModel



"""
Wrapper class for local inference
To use for local debugging:
    - Download .pth file of model for local storage 
    - 
"""
class ModelPredictor:
    def __init__(self, model: BinaryClassifierModel):
        self.model = model

    def run_local_inference(self, tensors: FloatTensor):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensors)  # generates raw, model output

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

    """
    Creates model from stored .pth file
    """
    @classmethod
    def from_path(cls, stored_model_file_path: Path):
        model = BinaryClassifierModel()
        model.load_state_dict(torch.load(stored_model_file_path))
        return cls(model)





