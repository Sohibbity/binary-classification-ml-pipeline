import torch

from ModelBuilding.BinaryClassifierModel import BinaryClassifierModel


class ModelRetriever():
    """
    Creates model from stored .pth file
    """
    @staticmethod
    def load_model(stored_model_file_path: str):
        model = BinaryClassifierModel()
        model.load_state_dict(torch.load(stored_model_file_path))
        return model
