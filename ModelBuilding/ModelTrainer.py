
from pandas import DataFrame
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim

from Config.Constants import Model_OUTPUT_PATH, Y_AXIS, X_AXIS
from DataModels.Tensors import Tensors
from ModelBuilding.BinaryClassifierModel import BinaryClassifierModel

class ModelTrainer:
    """
    Trains model
    Outputs model artifact to designated output
    Under /Models directory
    """
    def __init__ (self, model: BinaryClassifierModel):
        self.model = model

    def train_model(self, processed_df: DataFrame, model_output_path = Model_OUTPUT_PATH):

        # Assign test/train split and create tensors
        tensors = self._generate_tensors(processed_df)

        print(f"Training samples: {tensors['x_train_tensor'].shape}")
        print(f"Test samples: {tensors['x_test_tensor'].shape}")

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 20
        for epoch in range(num_epochs):
            outputs = self.model(tensors['x_train_tensor'])
            loss = criterion(outputs, tensors['y_train_tensor'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Save model

        torch.save(self.model.state_dict(), model_output_path)
        print(f"Model saved to {model_output_path}")

    def _generate_tensors(self, processed_df: DataFrame):
        x = processed_df.drop(Y_AXIS, axis=1)
        y = processed_df[Y_AXIS]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    
        tensors= Tensors(
            x_train_tensor = torch.FloatTensor(x_train.values),
            x_test_tensor=torch.FloatTensor(x_test.values),
            y_train_tensor=torch.LongTensor(y_train.values),
            y_test_tensor=torch.LongTensor(y_test.values)
        )
        return tensors


