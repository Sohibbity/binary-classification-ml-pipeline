import torch.nn as nn

from Config.Constants import DF_ROWS


class BinaryClassifierModel(nn.Module):
    """
    Builds a blank BinaryClassification Model that uses 2 layer MLP
    Note this is a blank model
    In order to Create a model, a training loop must be run, and model weights must be saved
    """

    def __init__(self, input_size: int = DF_ROWS):
        super(BinaryClassifierModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x