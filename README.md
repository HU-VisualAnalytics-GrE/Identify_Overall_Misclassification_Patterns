# Semester Project Group-E

## Development and Implementation Documentation
Look at the [Local Installation Instructions](docs/local-install-instructions.md) for further information on how to run the environment.

## Purpose and scope
This project aims to build an interactive system for analyzing and visualizing a Random Forest classification model for organic carbon prediction using dataset features. The system includes the following tasks and corresponding solutions:

Tasks and Solutions:

Data Preprocessing:

Task: Load and preprocess datasets.
Solution: The lucas_organic_carbon_target.csv and lucas_organic_carbon_training_and_test_data.csv files are read into pandas dataframes. Features (X) and target (y) variables are extracted, and the data is split into training and test sets.
Model Training and Evaluation:

Task: Train a Random Forest Classifier and evaluate its performance.
Solution: The RandomForestClassifier from scikit-learn is used to train the model on the training set. The model's predictions are compared to the test set labels, and classification performance is evaluated using accuracy, confusion matrix, and classification report.
Confusion Matrix Visualization:

Task: Visualize the confusion matrix of the model's predictions.
Solution: The confusion matrix is computed and visualized using Plotly, both in a raw and normalized form. A checkbox allows the user to toggle between the two views.
Principal Component Analysis (PCA):

Task: Visualize the dataset's high-dimensional features using PCA.
Solution: PCA is applied to reduce the dataset to two dimensions. A scatter plot is then created to visualize the true vs. predicted labels, with color coding to differentiate between true positives, false positives, and false negatives.
Interactive Features:

Task: Allow interactive exploration of misclassifications and PCA results.
Solution: The application offers a sidebar where users can select the true and predicted labels to filter the PCA plot. This allows users to focus on specific misclassifications and analyze the results.

## Components
Major Components of the VA System:
Data Loading and Preprocessing:

Status: Done
Description: The necessary CSV files are read and processed, and features and labels are extracted for training and testing.
Random Forest Classifier:

Status: Done
Description: The Random Forest model is trained, evaluated, and used for predictions on the test set.
Confusion Matrix Visualization:

Status: Done
Description: The confusion matrix is generated and displayed using Plotly with interactive features for normalizing the matrix.
PCA Visualization:

Status: Done
Description: PCA is applied to reduce the data to two dimensions, and an interactive scatter plot is generated for visualizing the misclassifications.
Interactive Sidebar:

Status: Done
Description: The sidebar allows users to select different true and predicted labels to filter the PCA plot and explore misclassifications.

## Major Implementation Activities
Data Loading and Preprocessing:

Status: Done
Description: The necessary CSV files are read and processed, and features and labels are extracted for training and testing.
Random Forest Classifier:

Status: Done
Description: The Random Forest model is trained, evaluated, and used for predictions on the test set.
Confusion Matrix Visualization:

Status: Done
Description: The confusion matrix is generated and displayed using Plotly with interactive features for normalizing the matrix.
PCA Visualization:

Status: Done
Description: PCA is applied to reduce the data to two dimensions, and an interactive scatter plot is generated for visualizing the misclassifications.
Interactive Sidebar:

Status: Done
Description: The sidebar allows users to select different true and predicted labels to filter the PCA plot and explore misclassifications.
Major Implementation Activities
1. Data Preparation and Preprocessing:
Activity: Load the dataset and prepare it for model training and evaluation.
Status: Done
Details: The pandas library is used to read the CSV files. The features and target variables are extracted from the datasets, and the data is split into training and testing sets using train_test_split from scikit-learn.
2. Model Development and Training:
Activity: Train the Random Forest model and evaluate its performance.
Status: Done
Details: The RandomForestClassifier is trained with a specified number of estimators and other hyperparameters. The performance is evaluated using accuracy, classification report, and confusion matrix.
3. Confusion Matrix Visualization:
Activity: Implement and visualize the confusion matrix.
Status: Done
Details: The confusion matrix is calculated using confusion_matrix from scikit-learn, and then visualized using Plotly with both raw and normalized views. The option to toggle between these two visualizations is provided.
4. PCA Implementation and Visualization:
Activity: Perform PCA for dimensionality reduction and visualize the results.
Status: Done
Details: PCA is implemented using the PCA class from sklearn.decomposition. The reduced data is visualized as a scatter plot using Plotly, allowing users to explore misclassifications by selecting specific true and predicted labels from the sidebar.
5. Interactive User Interface:
Activity: Create an interactive interface for exploring the results.
Status: Done
Details: Streamlit is used to create an interactive UI, including the sidebar for selecting label combinations for PCA visualization. The application allows users to explore the results interactively by selecting labels and toggling the confusion matrix normalization.
