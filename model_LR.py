from sklearn.linear_model import LinearRegression
from model_Base import BaseModel
from sklearn.metrics import r2_score

class LinearModel(BaseModel):
    def __init__(self, filepath):
        """Initializing the base model and setting up the linear model."""
        
        super().__init__(filepath)
        self.model = None
        self.train()

    def train(self):
        """Trains the linear regression model"""
        self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)

 
    def predict(self, x):
        """Predicts target values using the trained linear regression model"""
        
        if self.model is None:
            raise ValueError("Model has not been trained. Call the train method first.")
        
        x_preprocessed = self.preprocess_new_data(x)
        return self.model.predict(x_preprocessed)
    
    def print_results(self):
        """Prints the performance metrics of the model for CV and test sets."""
        
        # Predictions on test set
        y_pred_test = self.model.predict(self.x_test)
        test_score = r2_score(self.y_test, y_pred_test)
        
        # Predictions on CV set
        y_pred_cv = self.model.predict(self.x_cv)
        cv_score = r2_score(self.y_cv, y_pred_cv)

        # Print scores
        print(f"Linear Model Score (CV): {cv_score * 100:.2f}%")
        print(f"Linear Model Score (Test Set): {test_score * 100:.2f}%")
