from typing import TypedDict

import torch



class Tensors(TypedDict):
    """
    Some form of type safety for storing model tensors
    """
    x_train_tensor: torch.FloatTensor
    x_test_tensor: torch.FloatTensor
    y_train_tensor: torch.LongTensor
    y_test_tensor: torch.LongTensor



