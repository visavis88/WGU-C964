from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from model_Base import BaseModel

class RandomForestModel(BaseModel):

    def __init__(self, filepath):
        """Initialize the RandomForest model with default hyperparameters."""

        super().__init__(filepath)
        
        # this has been found to be the best configuration through hyperparameter_tuning()
        self.default_params = {
            'bootstrap': False,
            'criterion': 'friedman_mse',
            'max_depth': 20,
            'max_features': 'sqrt',
            'min_samples_leaf': 2,
            'min_samples_split': 10,
            'n_estimators': 100,
            'random_state': 1  # for reproducibility
        }
        
        # Initializing best_params with default parameters
        self.best_params_ = self.default_params

        
        self.model = RandomForestRegressor(**self.default_params)
        
        self.train()

    def train(self):
        """Train the model on the provided data."""

        self.model.fit(self.x_train, self.y_train)


    def hyperparameter_tuning(self):
        """Tune hyperparameters using GridSearchCV."""

        # Define the hyperparameters and their possible values
        param_grid = {
            'criterion': ['friedman_mse','mse', 'mae'],
            'n_estimators': [10, 50, 100, 200],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, 
                                   cv=3, n_jobs=-1, verbose=0)

        # Fit the model
        grid_search.fit(self.x_train, self.y_train)

        # Update the model with the best hyperparameters
        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        print(f"Best Parameters: {grid_search.best_params_}")

    def predict(self, x_new):
        """Make predictions based on new input data."""
    
        x_new_processed = self.preprocess_new_data(x_new)
        
        # Get predictions using the random forest model
        predictions = self.model.predict(x_new_processed)
        
        return predictions


    

    def print_results(self):
        """Print model scores on CV and test datasets."""

        # Predictions on test set
        y_pred_test = self.model.predict(self.x_test)
        test_score = r2_score(self.y_test, y_pred_test)
        
        # Predictions on CV set
        y_pred_cv = self.model.predict(self.x_cv)
        cv_score = r2_score(self.y_cv, y_pred_cv)

        # Print scores
        print(f"Random Forest Model Score (CV): {cv_score * 100:.2f}%")
        print(f"Random Forest Score (Test Set): {test_score * 100:.2f}%")
        


    def print_feature_importance(self):

        print("\n\nThe most important features are:")
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]
        variables = ['age', 'sex', 'bmi', 'children','smoker', 'region']
        importance_list = []
        for f in range(self.x.shape[1]):
            variable = variables[indices[f]]
            importance_list.append(variable)
            print("%d.%s(%.2f%%)" % (f + 1, variable, importances[indices[f]] * 100))
    

    def plot_feature_importance(self):
        """Plot a bar chart showing the importance of each feature."""
        
        # Get the importances and standard deviations
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        variables = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        
        # Plot the feature importances
        plt.figure(figsize=(8, 4))
        plt.title("Feature Importances")
        plt.bar(range(self.x.shape[1]), importances[indices], yerr=std[indices], align="center", color="lightblue", edgecolor="black")
        plt.xticks(range(self.x.shape[1]), [variables[i] for i in indices], rotation=45)
        
        # Adjust y-axis to display percentages
        y_vals = plt.gca().get_yticks()
        plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in y_vals])
        
        plt.xlim([-1, self.x.shape[1]])
        plt.tight_layout()
        plt.show()



