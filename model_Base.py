import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class BaseModel:
    def __init__(self, filepath):
        
        # Load dataset from provided filepath
        self.dataset = pd.read_csv(filepath)
       
        # Create a backup of the original dataset
        self.dataset_original = self.dataset.copy()
        
        # Initializing instance-level variables for data and data splits
        self.x = None
        self.y = None
        self.x_train = None
        self.x_cv = None
        self.x_test = None
        self.y_train = None
        self.y_cv = None
        self.y_test = None
        
        # Dictionaries to store StandardScaler and LabelEncoder objects
        self.scalers = {}
        self.encoders = {}
        
        # Start data preprocessing (encoding, splitting, scaling)
        self.preprocess_data()

    def preprocess_data(self):
        """  Main preprocessing function that encodes categorical features, splits the data, and scales the features. """
        self.encode_categorical()
        self.split_data()
        self.feature_scaling()

    def encode_categorical(self):
        """  Encodes categorical columns in the dataset using the LabelEncoder. """
        
        # Convert given columns to 'category' dtype
        self.dataset[['sex', 'smoker', 'region']] = self.dataset[['sex', 'smoker', 'region']].astype('category')
        
        # Loop through columns and encode them using LabelEncoder
        for col in ['sex', 'smoker', 'region']:
            label = LabelEncoder()
            
            # Store each encoder in the dictionary for potential use later
            self.encoders[col] = label  
            self.dataset[col] = label.fit_transform(self.dataset[col])

    def split_data(self):
        """ Splits the data into training, cross-validation, and test sets using a 70-15-15 split. """
        # Separate features and target variable
        self.x = self.dataset.drop('charges', axis=1).values
        self.y = self.dataset['charges'].values
        # Split data into training and temporary sets (70-30 split)
        self.x_train, x_temp, self.y_train, y_temp = train_test_split(self.x, self.y, train_size=0.7, random_state=42)
        # Further split the temporary set into cross-validation and test sets (50-50 split)
        self.x_cv, self.x_test, self.y_cv, self.y_test = train_test_split(x_temp, y_temp, train_size=0.5, random_state=42)



    def feature_scaling(self):
        """ Applies feature scaling (standardization) on training, cross-validation, and test data.  """
        
        scaler = StandardScaler()
        
        # Store the scaler object for potential inverse scaling later
        self.scalers['x'] = scaler
        
        
        # Convert DataFrame slices to numpy arrays before scaling
        self.x_train = scaler.fit_transform(self.x_train.astype(float))  # Fit and transform training data
        self.x_cv = scaler.transform(self.x_cv.astype(float))
        self.x_test = scaler.transform(self.x_test.astype(float))

        # Transform (using training data mean and std dev) cross-validation and test data
        #self.x_train = scaler.fit_transform(self.x_train)  # Fit and transform training data
        #self.x_cv = scaler.transform(self.x_cv)
        #self.x_test = scaler.transform(self.x_test)

    def train(self):
        # Placeholder for the training method to be implemented by subclasses
        raise NotImplementedError("Train method not implemented")

    def predict(self, x):
        # Placeholder for the prediction method to be implemented by subclasses
        raise NotImplementedError("Predict method not implemented")

    def preprocess_new_data(self, x):
        """Preprocess new input data in the same way as the training data."""
        df = pd.DataFrame(x, columns=self.dataset_original.drop('charges', axis=1).columns)
        
        # Encode categorical columns
        for col, encoder in self.encoders.items():
            df[col] = encoder.transform(df[col])
        
        # Scale the features
        x_scaled = self.scalers['x'].transform(df)
        return x_scaled
    

    def inverse_transform(self, x, y):
        """ 
        Inverse transforms feature scaled and label encoded data. 
        Helpful for visualization or further processing on original scale. 
        """
        
        x = pd.DataFrame(self.scalers['x'].inverse_transform(x), columns=self.dataset_original.drop('charges', axis=1).columns)
        
        # Loop through encoded columns and inverse transform them
        for col, encoder in self.encoders.items():
            x[col] = encoder.inverse_transform(x[col].astype(int))
        return x, y

    def plot_features_vs_target(self):
        """
        Plots various features against the target variable to understand their relationship.
        """
        
        # Define subplots layout
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        
        # Plot using the original (non-encoded, non-scaled) dataset
        sns.barplot(x='region', y='charges', hue='smoker', data=self.dataset_original, ax=axs[0, 0])
        axs[0, 0].set_title('Region vs Charges')
        sns.barplot(x='children', y='charges', hue='smoker', data=self.dataset_original, ax=axs[0, 1])
        axs[0, 1].set_title('Children vs Charges')
        sns.scatterplot(x='age', y='charges', hue='smoker', data=self.dataset_original, ax=axs[1, 0])
        axs[1, 0].set_title('Age vs Charges')
        sns.scatterplot(x='bmi', y='charges', hue='smoker', data=self.dataset_original, ax=axs[1, 1])
        axs[1, 1].set_title('BMI vs Charges')
        
        # Set main title for all plots
        fig.suptitle("Dataset Analysis", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_dataset_split_distribution(self):
        """
        Plots the distribution of the dataset after splitting into training, cross-validation, and test sets.
        Aids in understanding the distribution of data across the splits.
        """
        
        # Inverse transform data splits for visualization
        df_X_train, df_y_train = self.inverse_transform(self.x_train, self.y_train)
        df_X_cv, df_y_cv = self.inverse_transform(self.x_cv, self.y_cv)
        df_X_test, df_y_test = self.inverse_transform(self.x_test, self.y_test)
        
        
        # Convert numpy arrays to DataFrames for easier plotting
        df_y_train = pd.DataFrame(df_y_train, columns=['charges'])
        df_y_cv = pd.DataFrame(df_y_cv, columns=['charges'])
        df_y_test = pd.DataFrame(df_y_test, columns=['charges'])
        
        # Concatenate features and target to form complete datasets for each split
        df_train = pd.concat([df_X_train, df_y_train], axis=1)
        df_cv = pd.concat([df_X_cv, df_y_cv], axis=1)
        df_test = pd.concat([df_X_test, df_y_test], axis=1)
        
        # Define subplots layout
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        sns.scatterplot(x='age', y='charges', data=df_train, ax=axes[0])
        axes[0].set_title('Age vs Charges (Train)')
        sns.scatterplot(x='age', y='charges', data=df_cv, ax=axes[1])
        axes[1].set_title('Age vs Charges (CV)')
        sns.scatterplot(x='age', y='charges', data=df_test, ax=axes[2])
        axes[2].set_title('Age vs Charges (Test)')
        
        fig.suptitle("Dataset Split Distribution", fontsize=16)
        plt.tight_layout()
        plt.show()
