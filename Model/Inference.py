import json
import os
from typing import TypedDict

import torch

from Config.Constants import MODEL_V1
from Model.BinaryClassifierModel import BinaryClassifierModel

class PredictionResult(TypedDict):
    probabilities: torch.Tensor
    predictions: torch.Tensor
    confidences: torch.Tensor


def model_fn(model_dir,):
    model = BinaryClassifierModel(input_size = 8)
    model_path = os.path.join(model_dir,f'{MODEL_V1}.pth')
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    # As is for now, app side processes s3 csv chunk and passes list to the api call
    # Deserialize network request body, convert to list
    if request_content_type == 'application/json':
        # this is csv now, not a df
        # models dont touch csvs or dfs, their language ARE TENSORS
        deserialized_data_list = json.loads(request_body)
        x_tensors = torch.FloatTensor(deserialized_data_list)
        return x_tensors
    else:
        raise ValueError(f"Unsupported content type: {request_content_type} "
                         f"This endpoint currently only supports JSON")

def predict_fn(input_data, model) -> PredictionResult:
    # Determine device from where the model is
    device = next(model.parameters()).device

    # Move input data to same device as model
    input_data = input_data.to(device)

    with torch.no_grad():
        outputs = model(input_data)  # generates raw, model output

        probabilities = torch.softmax(outputs, dim=1)  # Convert to probabilities
        predictions = torch.argmax(probabilities, dim=1)  # Get predicted class (0 or 1)
        confidences = torch.max(probabilities, dim=1).values  # Get confidence score

        result: PredictionResult =  {
            'probabilities': probabilities,
            'predictions': predictions,
            'confidences': confidences,
        }

        return result

def output_fn(prediction_result, content_type):
    # convert tensors to float lists
    if content_type == 'application/json':
        return json.dumps({
            'probabilities': prediction_result['probabilities'].cpu().numpy().tolist(),
            'predictions': prediction_result['predictions'].cpu().numpy().tolist(),
            'confidences': prediction_result['confidences'].cpu().numpy().tolist(),
        })
    else:
        raise ValueError(f"Unsupported content type: {content_type} "
                         f"This endpoint currently only supports JSON")
