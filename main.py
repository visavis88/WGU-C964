from model_Base import BaseModel
from model_LR import LinearModel
from model_PR import PolynomialRegressionModel
from model_NN import NeuralNetworkModel
from model_RFR import RandomForestModel
from art import *
import sys
import os
from colorama import Fore, Style
import warnings




base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(base_dir, "insurance.csv")




# Text formatting
BOLD = Style.BRIGHT
END = Style.RESET_ALL

# Text colors
GREEN = Fore.GREEN
RED = Fore.RED
PURPLE = Fore.MAGENTA
YELLOW = Fore.YELLOW
BLUE = Fore.CYAN


# Pre-formatted strings
str_separator = f"\n{PURPLE}================================================{END}\n"
str_pressEnter = f"\n\n{BLUE}Press ENTER to continue...{END}"
str_choiceSelection = f"{BLUE}Enter your choice: {END}"


# Test dataset, not seen during training
test_dataset = [
    [25, 'male', 26.22, 0, 'no', 'northeast', 2721.3208],
    [64, 'female', 39.33, 0, 'no', 'northeast', 14901.5167],
    [30, 'female', 19.95, 3, 'no', 'northwest', 5693.4305],
    [57, 'female', 23.98, 1, 'no', 'southeast', 22192.43711],
    [31, 'male', 27.645, 2, 'no', 'northeast', 5031.26955],
    [22, 'male', 52.58, 1, 'yes', 'southeast', 44501.3982],
    [32, 'female', 41.1, 0, 'no', 'southwest', 3989.841],
    [33, 'male', 35.75, 1, 'yes', 'southeast', 38282.7495],
    [28, 'male', 31.68, 0, 'yes', 'southeast', 34672.1472],
    [43, 'female', 35.64, 1, 'no', 'southeast', 7345.7266],
    [49, 'male', 28.69, 3, 'no', 'northwest', 10264.4421],
    [41, 'male', 23.94, 1, 'no', 'northeast', 6858.4796],
    [30, 'male', 25.46, 0, 'no', 'northeast', 3645.0894],
    [48, 'female', 33.33, 0, 'no', 'southeast', 8283.6807],
    [35, 'female', 35.86, 2, 'no', 'southeast', 5836.5204],
    [56, 'female', 35.8, 1, 'no', 'southwest', 11674.13],
    [25, 'female', 32.23, 1, 'no', 'southeast', 18218.16139]
]

warnings.filterwarnings("ignore") #suppress warnings


def menu_analyze_raw_data():
    """Menu for analyzing raw dataset."""

    # ASCII art for the title
    tprint("Raw Data Analysis")
    
    print("📊 The following plot will help visualize and analyze the relationship between various features (independent variables) and the target variable (dependent variable) in the dataset. \nIt provides insights into how different features influence the target variable.")
    input(str_pressEnter)

    base = BaseModel(filepath)
    base.plot_features_vs_target()

    
    print("\n\n📊 The following plot will help visualize the distribution of the dataset after it has been split into training, cross-validation, and test sets. \nIt aids in understanding how the data is distributed across these splits.")
    input(str_pressEnter)

    base = BaseModel(filepath)
    base.plot_dataset_split_distribution()
            

def menu_training_results():
    """Menu to display model training results."""
    
    print(str_separator)
    tprint("Training Results")

    # Linear Regression
    model_linear.print_results()
    input(str_pressEnter)

    # Polynomial Regression
    print(str_separator)
    model_poly.print_results()

    print("\n📊 The following plots represents the performance metrics of polynomial models against various polynomial degree and alphas.\n")
    input(str_pressEnter)
    
    model_poly.plot_results()
    input(str_pressEnter)

    # Neural Networks
    print(str_separator)
    model_nn.print_results()
    
    print("\n📊 The following plot represents the performance metrics of Neural Network during training.\n")
    input(str_pressEnter)
    
    model_nn.plot_results()
    input(str_pressEnter)

    # Random Forest
    print(str_separator)
    model_forest.print_results()
    
    print("\n📊 The following plot represents how much each feature influences the cost of insurance.\n")
    input(str_pressEnter)
    
    model_forest.plot_feature_importance()
    input(str_pressEnter)


def print_predictions(predictions, model_name, target=None):
    """Display the predicted charges in more readable format."""

    print(f"\nPredicted Charges ({GREEN}{model_name}{END})")
    for i, prediction in enumerate(predictions):
        formatted_prediction = f"{YELLOW}${prediction:.2f}{END}"
        if target is not None:
            formatted_target = f"{YELLOW}${target[i]:.2f}{END}"
            formatted_difference = f"{YELLOW}${abs(target[i]-prediction):.2f}{END}"
            print(f" {i + 1}: Predicted: {formatted_prediction}, \t Actual: {formatted_target} \t Difference: {formatted_difference}")
        else:
            print(f" {i + 1}: {formatted_prediction}")


def models_performance_comparison(x_pred = [], target = None):
    """Compare performance of different models."""

    if x_pred is None or len(x_pred) == 0:
        x_pred = [row[:-1] for row in test_dataset]
        target = [row[-1] for row in test_dataset]
    
    print(str_separator)
    predictions_linear = model_linear.predict(x_pred)
    predictions_poly = model_poly.predict(x_pred)
    predictions_nn = model_nn.predict(x_pred)
    predictions_forest = model_forest.predict(x_pred)


    print_predictions(predictions_linear, "linear", target)
    print_predictions(predictions_poly, "poly", target)
    print_predictions(predictions_nn, "nn", target)
    print(f"\n\n{BOLD}Recommended model: {END}")
    print_predictions(predictions_forest, "forest", target)



    input(str_pressEnter)
   

def menu_prediction():
    """Menu for prediction options."""
    while True:
        print(str_separator)
        tprint("Prediction Menu")
        print("1. Use Sample Data")
        print("2. Enter Your Own Data")
        print("0. Back to Main Menu\n")

        choice = input(str_choiceSelection)

        if choice == '1':
            models_performance_comparison()
        elif choice == '2':
            while True:
                
                print(f"\n{BOLD}Format {END}: {YELLOW} age, sex, bmi, children, smoker, region {END} (include commas between values)...")
                print(f"{BOLD}Example {END}: 30, male, 25.5, 2, no, southwest\n")
                input_str = input(f"{BLUE}Enter data (or '0' to go back to the previous menu): {END}")

                if input_str.lower() == '0':
                    break

                data = input_str.split(',')
                

                str_invalidInput = f"{RED}Invalid input! {END}"
                if len(data) != 6:
                    print(f"{str_invalidInput} Please enter data in the correct format.")
                    continue

                try:
                    age = int(data[0])
                    sex = data[1].strip().lower()
                    bmi = float(data[2])
                    children = int(data[3])
                    smoker = data[4].strip().lower()
                    region = data[5].strip().lower()

                    # Perform checks on the data
                    if age <= 0:
                        print(f"{str_invalidInput} Age must be a positive integer.")
                        continue
                    if sex not in ['male', 'female']:
                        print(f"{str_invalidInput} Sex must be 'male' or 'female'.")
                        continue
                    if bmi <= 0:
                        print(f"{str_invalidInput} BMI must be a positive number.")
                        continue
                    if not (0 <= children <= 10):
                        print(f"{str_invalidInput} Children must be an integer between 0 and 10.")
                        continue
                    if smoker not in ['yes', 'no']:
                        print(f"{str_invalidInput} Smoker must be 'yes' or 'no'.")
                        continue
                    if region not in ['southwest', 'southeast', 'northwest', 'northeast']:
                        print(f"{str_invalidInput} Region must be 'southwest', 'southeast', 'northwest', or 'northeast'.")
                        continue

                    # Data is valid, use it for prediction
                    x_pred = [[age, sex, bmi, children, smoker, region]]
                      
                    models_performance_comparison(x_pred)

                    
                    
                except ValueError as e:
                    print(f"Error: {e}")
                    print("Invalid input. Please enter data in the correct format.")
                    continue

        elif choice == '0':
            break
        else:
            print("Invalid choice. Please try again.")
    
   
def main_menu():
    """Main menu for the program."""
    while True:
        tprint("Main Menu")
        print("1. Analyze Raw Data")
        print("2. Show the Models' Training Results")
        print("3. Predict the Cost of Insurance")
        print("0. Exit\n")

        choice = input(str_choiceSelection)

        if choice == '1':
            menu_analyze_raw_data()
        elif choice == '2':
            menu_training_results()
        elif choice == '3':
            menu_prediction()
        elif choice == '0':
            break
        else:
            print("Invalid choice. Please try again.")








# =========================================================================
#                      Program's Starting Point
# =========================================================================



tprint("Hello")

print(
    f"{GREEN}{BOLD}👋 Welcome to the Intelligent Premium Estimator!{END}\n\n"
    f"In the upcoming steps, we'll be training four different machine learning models to predict insurance costs:\n"
    f"\t{BOLD}1. Linear Regression Model{END}\n"
    f"\t{BOLD}2. Polynomial Regression Model{END}\n"
    f"\t{BOLD}3. Neural Network Model{END}\n"
    f"\t{BOLD}4. Random Forest Regression Model{END}\n\n"
    f"Our goal is to evaluate the performance of each model and determine which one provides the most accurate predictions.\n"
    f"After training, you'll be able to view detailed results and even predict insurance costs based on custom inputs.\n\n"
    f"{BOLD}Let's dive in!{END}"
)



choice = input(str_pressEnter)
print("Please wait while we train our models...\n\n")

# train our models
model_linear = LinearModel(filepath)
model_poly = PolynomialRegressionModel(filepath)
model_nn = NeuralNetworkModel(filepath)
model_forest = RandomForestModel(filepath)


main_menu()