import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

from model_Base import BaseModel


class NeuralNetworkModel(BaseModel):
    
    def __init__(self, filepath):
        """Initialize the NeuralNetwork model with a default architecture."""

        super().__init__(filepath)
        self.model = Sequential(
            [
                Dense(64, activation='relu',  name="L1"),
                Dense(128, activation='relu', name="L2"),
                Dense(64, activation='relu', name="L3"),
                Dense(1, activation='linear', name="L_Output"),
            ],
            name="default_model"
        )
        
        self.train()

        

    def train(self):
        """Train the neural network using Adam optimizer and mean squared error loss."""

        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=100,
            batch_size=16,
            validation_data=(self.x_cv, self.y_cv),
            verbose=0
        )

    def plot_results(self):
        """Plot training and validation loss over epochs."""

        plt.figure(figsize=(9, 5))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.show()

    def predict(self, x_new):
        """Predict the output for new data using the trained neural network."""

        # Preprocess the new data
        x_new_processed = self.preprocess_new_data(x_new)
        
        # Get predictions using the neural network model
        predictions = self.model.predict(x_new_processed)
    
        return predictions.ravel()


    def print_results(self):
        """Print model performance on CV and test datasets."""
        
        # Predictions on test set
        y_pred_test = self.model.predict(self.x_test)
        test_score = r2_score(self.y_test, y_pred_test)
        
        # Predictions on CV set
        y_pred_cv = self.model.predict(self.x_cv)
        cv_score = r2_score(self.y_cv, y_pred_cv)

        # Print scores
        print(f"Neural Network Model Score (CV): {cv_score * 100:.2f}%")
        print(f"Neural Network Model Score (Test Set): {test_score * 100:.2f}%")
   