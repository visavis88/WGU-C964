# WGU-C964
Capstone Project

# Project: Healthcare Cost Prediction

## Solution Summary

- The project addressed a critical challenge in the healthcare sector: the prediction of healthcare costs based on various patient attributes. This issue, complex due to the variable nature of individual health data and the diverse range of factors influencing healthcare costs, demanded a solution that was both adaptable and precise.

- Our application served as a comprehensive solution, deploying machine learning to predict potential healthcare costs with significantly increased accuracy. We designed and implemented a multi-model approach that included Linear Regression, Polynomial Regression, Neural Networks, and Random Forest Regression. The Linear Regression model was used for setting a performance benchmark for subsequent, more complex models.

- The application provided an effective solution by allowing users, particularly healthcare providers and individuals preparing for healthcare expenses, to input specific patient attributes and receive a precise prediction of potential healthcare costs. This was instrumental in financial planning and resource allocation for both providers and patients.

- Through the integration of these models, our system not only predicted costs but also helped users understand the influential factors, thereby offering insights into areas for cost management and reduction. This was particularly evident in the case of the Random Forest model, which highlighted the most significant variables contributing to healthcare costs.

## Data Summary

- The raw data utilized for this project was sourced from a publicly available healthcare cost dataset on Kaggle, accessible at Kaggle Medical Insurance Costs Dataset. The dataset included diverse patient information such as age, sex, BMI, number of children, smoking status, region, and charges, all critical factors in predicting insurance costs. Notably, the dataset was comprehensive and well-compiled, with no missing values, which streamlined the initial data preparation stages, allowing the team to focus on more intricate aspects of data processing.

- Our approach to handling this data was encapsulated in the design of the 'BaseModel' class, a blueprint for the subsequent specific model classes. One of the class's initial steps was to create a backup of the original dataset, ensuring that there was a fallback option and reference point as the data underwent various transformations.

- The 'BaseModel' class was integral in automating the data processing tasks. It began with the encoding of categorical variables, such as 'sex,' 'smoker,' and 'region,' using the LabelEncoder from the scikit-learn library. This step converted textual data into a numerical format, a prerequisite for machine learning algorithms. Each encoder was meticulously preserved for future inverse transformations, ensuring consistency and the ability to revert to original data states.

- Subsequently, the data was divided into training, cross-validation, and testing sets, adopting a 70-15-15 split to ensure ample data for training and sufficient subsets for validation and testing. This strategic division safeguarded the model's ability to generalize well to unseen data, an essential characteristic given the diverse potential inputs in practical application scenarios.

- Feature scaling, implemented via StandardScaler, was the next pivotal step, normalizing the feature set to standardize the range of independent variables. This process was crucial for algorithms dependent on distance calculations and helped to expedite the training phase.

- Moreover, we included methods for handling new data inputs (preprocess_new_data), ensuring they underwent the same transformation pipeline as the original dataset. This consistency was vital for the model's accuracy, as deviation in data processing could significantly impair prediction reliability.

- The BaseModel class was not just a workhorse for data handling but also served as an analytical tool. It included methods for plotting relationships between features and the target variable ('plot_features_vs_target'), utilizing the original dataset to highlight trends and patterns pre-encoding and scaling. Furthermore, the class could visualize the distribution of data across training, cross-validation, and test sets ('plot_dataset_split_distribution'), ensuring transparency and understanding of the dataset's composition.

- Overall, the 'BaseModel' class formed the backbone of the application's data management strategy, ensuring a systematic, consistent, and transparent approach to data transformation, leading to reliable, accurate predictions in healthcare cost estimation.

## Machine Learning

The application developed here is intricately designed to predict healthcare costs, relying on various patient attributes such as age, BMI, and medical history. This integration of machine learning is crucial, offering precise cost predictions essential for individuals and healthcare providers preparing for financial aspects of healthcare services. The machine learning component, central to this application, utilizes sophisticated algorithms to process historical data, identifying patterns that contribute significantly to healthcare costs. Python's versatile libraries, including Scikit-learn for model building and Matplotlib for visual insights, were instrumental in this developmental phase.

Addressing the unpredictable nature of healthcare data and the necessity for precision in cost prediction, we adopted a multi-model approach. We explored various models, namely Linear Regression, Polynomial Regression, Neural Networks, and Random Forest Regression, due to initial uncertainties in data trends and the complexities involved in accurate cost prediction. This comprehensive approach ensured a robust analysis from multiple perspectives, enhancing the reliability of the predictions.

### BaseModel Class

- The foundation for our models is established by the BaseModel class. This class, aside from loading and maintaining a copy of the original dataset, plays a pivotal role in preparing the data for training and prediction. The preprocessing pipeline embedded within BaseModel involves encoding categorical data with the encode_categorical() method, followed by dividing the dataset into training, cross-validation, and testing subsets via the split_data() function. After this partitioning, each subset undergoes feature scaling using the feature_scaling() method. The class also presents methods to prepare new data inputs for prediction (preprocess_new_data()) and to reverse the encoding and scaling transformations (inverse_transform()), which proves beneficial for data visualization or analysis on the original data scale. Furthermore, to facilitate the visual exploration of the raw dataset, BaseModel integrates two distinct plotting functions. The plot_features_vs_target() method provides insights into the relationship between various features and the target variable, whereas the plot_dataset_split_distribution() function displays the data distribution across training, cross-validation, and testing sets.

### Linear Regression Model

- Linear Regression, a fundamental technique in predictive modeling, was our first logical step due to its interpretability and capability of capturing the primary trend in data with minimal complexity. This method, assuming a linear relationship between the target and predictor variables, serves as a baseline model from which we could compare the complexities and accuracies of more advanced models.

- Implemented through the LinearRegression class from the sklearn library, the training phase involves fitting the model with the training data, followed by making predictions using the predict() function. This straightforward approach was essential for setting a performance benchmark for subsequent, more complex models.

### Polynomial Regression Model

- The PolynomialRegressionModel class is an advanced regression model derived from the base model BaseModel. This model leverages polynomial features and the option of Ridge regularization to provide more flexibility and potentially better performance than a simple linear regression. Key features of this model include:

### Neural Network Model

- The NeuralNetworkModel class incorporates the modern deep learning approach, using TensorFlow to develop neural network architectures capable of capturing intricate data patterns. This model is constructed sequentially, layering neurons to understand data nuances deeply.

- Upon initialization, the neural network architecture is a composition of three dense layers. These layers consist of 64, 128, and 64 neurons, each activated by the ReLU function, known for its efficacy in introducing non-linearity. Culminating this network is the output layer with a solitary neuron, embodying a linear activation, suited for regression tasks.

- The train() method is pivotal to the model's learning. It employs the Adam optimizer, renowned for its efficiency and adaptive learning rates. The neural network is optimized to minimize the mean squared error, a standard loss metric for regression problems. The model's training process is refined through several iterations (epochs), with the number chosen based on empirical evidence ensuring effective learning. The batch size was also optimized according to standard practices, balancing the computational load and the model's ability to generalize from the training data.

- For a vivid representation of the training journey, the plot_results() method elegantly illustrates the ebbs and flows of both training and validation losses throughout the epochs. This visualization serves as an essential tool in deciphering the model's learning behavior and diagnosing issues like overfitting.

- The class also accommodates prediction capabilities through the predict() method. It seamlessly preprocesses new data and utilizes the trained neural network to produce the required predictions.

- In the end, the print_results() method provides a comprehensive overview of the model's prowess. It furnishes the r-squared score, a statistic that measures the goodness of fit, on both the test and cross-validation datasets. This score is invaluable in understanding the model's ability to explain the variance in the target variable.

### Random Forest Regression Model

- The RandomForestModel class is designed to harness the powerful ensemble technique of Random Forest regression. This model thrives by combining multiple decision trees to produce a more accurate and stable prediction.

- Upon initialization, the model is configured with a set of default hyperparameters, discovered to be the most optimal through prior invocation of hyperparameter_tuning(), which employs the GridSearchCV approach. This method undertakes an exhaustive search over multiple combinations of hyperparameter values. The grid encompasses varying criterion types, the number of trees (n_estimators), maximum features to consider for splits, tree depth (max_depth), and criteria for leaf node splits. After performing this comprehensive search, the model updates its hyperparameters based on the best-performing combination.

- A significant advantage of Random Forest is its ability to gauge the importance of each feature. The print_feature_importance() method capitalizes on this by ranking features according to their significance in prediction. This offers valuable insights, especially in understanding which variables hold the most weight in the prediction process.

- For a more visual interpretation, the plot_feature_importance() method provides a graphical representation of these importances. This method uses error bars to portray the variability in importance values across the multiple trees in the ensemble, offering a more intuitive understanding of the model's inner workings.

In conclusion, each chosen model plays a strategic role in this comprehensive approach to predicting healthcare costs. Linear Regression establishes a foundational understanding, while Polynomial Regression introduces an advanced level of nuance in interactions within the data. Neural Networks delve deeper into the intricacies, capturing complex patterns and relationships, whereas Random Forest Regression contributes robustness and an ensemble strategy for prediction. Together, these models form a cohesive framework, addressing the multifaceted nature of healthcare cost prediction. They not only stand on their individual strengths but also complement each other, ensuring that the system remains adaptable and efficient in accommodating the diverse and dynamic factors influencing healthcare costs.


## Validation

- BaseModel class was fundamental in setting up the data for training and testing, a critical first step for the subsequent assessment of each model's performance. It handled the splitting of the dataset, a method widely recognized as hold-out validation, into training, cross-validation, and testing subsets. This division is crucial for unbiased performance evaluation.

- LinearModel class, an extension of the BaseModel, simplified the process of linear regression. Upon model training completion, we employed the R-squared metric, a standard in regression analysis, to scrutinize the model's effectiveness. This metric, applied to both the cross-validation and test sets, provided immediate feedback on the model's predictive power and generalization capabilities. While the model showed promising preliminary results, future steps include a more detailed analysis of residuals and potentially exploring alternative metrics that could provide additional insights into model performance.

- PolynomialRegressionModel class, designed for capturing non-linear relationships, introduced an advanced level of adaptability. This class automatically determined the optimal polynomial degree and regularization parameter to prevent overfitting. We used cross-validation, a robust validation technique, for hyperparameter tuning, ensuring the model's effectiveness on unseen data. The results were promising, as reflected by the R-squared values; however, we planned to conduct further tests to confirm these findings and determine if further refinement was necessary.

- NeuralNetworkModel class marked our foray into deep learning. With its multi-layer architecture, this model was adept at capturing complex data patterns. We used 'relu' activation functions for their efficiency in helping the network learn and avoid certain pitfalls in training. During development, we observed the model's learning curve through its loss reduction, a common practice in deep learning validation. The preliminary R-squared scores on the test sets were encouraging, indicating a satisfactory level of model accuracy. In the future, we intended to explore different architectures and training techniques to potentially enhance performance further.

- RandomForestModel class provided a robust approach to regression problems. Initiated with a set of optimal parameters, it further employed an exhaustive hyperparameter optimization process, which we made more accessible by describing it as a method to find the most effective model settings. We used cross-validation scores to select the best parameters, a strategy known for its balance between model performance and overfitting prevention. The model's initial results were positive, reflected in high R-squared scores, suggesting that the model was accurately capturing the data's variability. For future work, we planned an in-depth feature importance analysis to understand better which variables the model deemed most influential in predictions.


# User Guide

## Step 1: Setting Up the Environment
- Open your web browser and navigate to Google Colab.
- Sign in with your Google account (or create one if you don't have it already).

## Step 2: Opening the Provided Notebook
- Go to this GitHub page (https://github.com/visavis88/WGU-C964.git).
- Download the Jupyter notebook file and the data file:
  - [Intelligent Premium Estimator.ipynb]
  - [insurance.csv]
- Back in Google Colab, click on the File menu in the upper left corner, then select Upload notebook....
- Choose the downloaded Intelligent Premium Estimator.ipynb file from your computer to upload it.
- Once uploaded, the notebook will open in a new tab, ready for use.

## Step 3: Uploading the Dataset
- In Colab, click on the folder icon in the left sidebar.
- Click on the Upload icon (which looks like a file with an upward arrow), and select the insurance.csv file from your computer. Wait for the file to upload completely.

## Step 4: Running the Notebook
- The notebook you've uploaded is pre-filled with the necessary code. To run all cells in the notebook, click on Runtime in the top menu and then select Run all.
- Alternatively, you can manually run each cell by clicking on it and then pressing Shift + Enter. Start from the top and work your way down to ensure the code executes in order. Make sure to click on the first cell first, installing all the necessary packages for the program's virtual environment. It will look similar to this:


## Step 5: Interacting with the Model

### 1. Introduction:
- Upon starting the program, users are greeted with a welcome message and a brief overview of what the application offers: training different machine learning models to predict insurance costs. Press ENTER to proceed.

### 2. Main Menu:
- The main menu provides three options:
1. Analyze Raw Data
2. Show the Models' Training Results
3. Predict the Cost of Insurance
0. Exit (by pressing '0')
- Navigate by entering the corresponding number for each item.

### 3. Analyzing Raw Data (Option 1):
- This option allows users to visualize the raw data. It displays plots representing the relationships between various features and the target variable, and the distribution of data across training, cross-validation, and test sets. Press ENTER to navigate through the visualizations.

### 4. Viewing Training Results (Option 2):
- Users can review detailed training results from the four models. The results include performance metrics and visual representations such as graphs or feature importance charts, specific to each type of model. Press ENTER to continue scrolling through the results.

### 5. Predicting Insurance Costs (Option 3):
- The prediction menu offers two scenarios for predicting insurance costs:
- Use Sample Data: The program will predict insurance costs using pre-defined sample data, displaying the results for comparison.
- Enter Your Own Data: Users can input their own data in the specified format. User must provide the information in the following format: age, sex, bmi, children, smoker, region. The program offers guidance if the input format is incorrect or illogical based. After entering valid data, the program predicts the insurance costs using all trained models for a comprehensive comparison. Then it identifies the most accurate model based on past performances and highlights this prediction, offering the user what it deems to be the best estimate for their insurance costs.
- To go back to the previous menu, enter '0'.

### 6. Exiting the Program (Option 0):
- Users can exit the program by selecting '0' from the main menu.
