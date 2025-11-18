from datetime import datetime

import pandas as pd
import torch
from pandas import DataFrame

from Model.BinaryClassifierModel import BinaryClassifierModel


class ModelPredictor:
    def __init__(self, model: BinaryClassifierModel):
        self.model = model

    # Defensability Analysis:
    # this is local rn but imagine it was an endpoint
    # Issues that warrant a retry:
    # network con to endppint dropped
    # model cpu limit exceeded, (yes we're chunking but maybe this time our file has the same number of rows, BUT each row is more
    # densley occuipied i.e if we had a synopsis col, previously col avg chars was 100, but we didnt realize this file its 5k char each, would
    # blow up model cpu
    # model inference is idempotent (all or nothing) so we retry 3x or drop from the entire batch of results
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




