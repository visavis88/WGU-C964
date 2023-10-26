import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

from model_Base import BaseModel

class PolynomialRegressionModel(BaseModel):


    def __init__(self, filepath):
        """Initializing base model and setting hyperparameters."""

        super().__init__(filepath)
        
        self.alphas = [0.01, 0.1, 0.5, 1, 10, 100, 500, 1000]
        self.degrees = list(range(1, 5))

        #Lists to store errors and scores for different hyperparameters.
        self.train_errors_degree = []
        self.cv_errors_degree = []
        self.scores_degree = []
        
        self.train_errors_alpha = []
        self.cv_errors_alpha = []
        self.scores_alpha = []

        # Default values for best hyperparameters
        self.best_degree = 2
        self.best_alpha = 1
        self.best_model = None  
        self.poly = None

        self.train()


    def train(self):
        """Training logic to find best hyperparameters and initialize best model."""
        
        self.find_best_degree()
        self.find_best_alpha_ridge()
        self.initialize_best_model()


    def initialize_best_model(self):
        """Initialize the best model based on best hyperparameters found."""
       
        self.poly = PolynomialFeatures(degree=self.best_degree)
        x_train_pol = self.poly.fit_transform(self.x_train)
        
        if self.best_alpha == 0:
            self.best_model = LinearRegression()
        else:
            self.best_model = Ridge(alpha=self.best_alpha)
            
        self.best_model.fit(x_train_pol, self.y_train)

    
    def find_best_degree(self):
        """Find the best polynomial degree for the model."""
        
        for degree in self.degrees:
            # Following the provided code for polynomial regression
            poly = PolynomialFeatures(degree=degree)
            x_train_pol = poly.fit_transform(self.x_train)
            x_cv_pol = poly.transform(self.x_cv)
            x_test_pol = poly.transform(self.x_test)

            poly_reg = LinearRegression()
            poly_reg.fit(x_train_pol, self.y_train)

            y_train_pred = poly_reg.predict(x_train_pol)
            y_cv_pred = poly_reg.predict(x_cv_pol)

            train_error = mean_squared_error(self.y_train, y_train_pred)
            cv_error = mean_squared_error(self.y_cv, y_cv_pred)

            y_test_pred = poly_reg.predict(x_test_pol)
            score = r2_score(self.y_test, y_test_pred)
            self.train_errors_degree.append(train_error)
            self.cv_errors_degree.append(cv_error)
            self.scores_degree.append(score)

        self.best_degree = np.argmin(self.cv_errors_degree) + 1

    def find_best_alpha_ridge(self):
        """Find the best alpha value for Ridge regularization."""
        
        poly = PolynomialFeatures(degree=self.best_degree)
        x_train_pol = poly.fit_transform(self.x_train)
        x_cv_pol = poly.transform(self.x_cv)
        x_test_pol = poly.transform(self.x_test)

        for alpha in self.alphas:
            ridge_reg = Ridge(alpha=alpha)
            ridge_reg.fit(x_train_pol, self.y_train)

            y_train_pred = ridge_reg.predict(x_train_pol)
            y_cv_pred = ridge_reg.predict(x_cv_pol)
            train_error = mean_squared_error(self.y_train, y_train_pred)
            cv_error = mean_squared_error(self.y_cv, y_cv_pred)
            score = ridge_reg.score(x_test_pol, self.y_test)

            self.train_errors_alpha.append(train_error)
            self.cv_errors_alpha.append(cv_error)
            self.scores_alpha.append(score)

        self.best_alpha = self.alphas[np.argmin(self.cv_errors_alpha)]

    def predict(self, x_new):
        """Predict the output based on new input data using the best model."""
        
        # Preprocess the new data
        x_new_processed = self.preprocess_new_data(x_new)
        
        x_new_pol = self.poly.fit_transform(x_new_processed)
        
        # Predict using the best model
        predictions = self.best_model.predict(x_new_pol)
            
        return predictions

    def plot_results(self):
        """Plot the performance metrics vs hyperparameters."""
        
        # Get degrees and alphas
        degrees = self.degrees
        alphas = self.alphas

        fig, axs = plt.subplots(2, 2, figsize=(10, 7))

        # Adding a main title for the entire figure
        fig.suptitle("Analysis of Model's Performance", fontsize=16, y=1.08)


        # First Row - For degree
        axs[0, 0].plot(degrees, self.train_errors_degree, label="Train Error", marker='o')
        axs[0, 0].plot(degrees, self.cv_errors_degree, label="CV Error", marker='o')
        axs[0, 0].set_xlabel('Degree')
        axs[0, 0].set_ylabel('Error')
        axs[0, 0].set_title('Train & CV Error vs Degree')
        axs[0, 0].legend()

        axs[0, 1].plot(degrees, self.scores_degree, label="Score", marker='o')
        axs[0, 1].set_xlabel('Degree')
        axs[0, 1].set_ylabel('R^2 Score')
        axs[0, 1].set_title('Score vs Degree')
        axs[0, 1].legend()

        # Second Row - For alpha
        axs[1, 0].plot(alphas, self.train_errors_alpha, label="Train Error", marker='o')
        axs[1, 0].plot(alphas, self.cv_errors_alpha, label="CV Error", marker='o')
        axs[1, 0].set_xlabel('Alpha')
        axs[1, 0].set_ylabel('Error')
        axs[1, 0].set_title('Train & CV Error vs Alpha')
        axs[1, 0].legend()

        axs[1, 1].plot(alphas, self.scores_alpha, label="Score", marker='o')
        axs[1, 1].set_xlabel('Alpha')
        axs[1, 1].set_ylabel('R^2 Score')
        axs[1, 1].set_title('Score vs Alpha')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def print_results(self):
        """Print the performance of the model and best hyperparameters."""

        poly = PolynomialFeatures(degree=self.best_degree)
        x_cv_pol = poly.fit_transform(self.x_cv)
        x_test_pol = poly.fit_transform(self.x_test)
        
        y_cv_pred = self.best_model.predict(x_cv_pol)
        y_test_pred = self.best_model.predict(x_test_pol)

        print(f"Polynomial Model Score (CV): {r2_score(self.y_cv, y_cv_pred) * 100:.2f}%")
        print(f"Polynomial Model Score (Test): {r2_score(self.y_test, y_test_pred) * 100:.2f}%\n")
        print(f"Best degree: {self.best_degree}")
        print(f"Best alpha: {self.best_alpha}\n")
        
  



    
