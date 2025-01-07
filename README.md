## Development and Implementation Documentation
- Pyhthon3.12 is recommended
- lucas_organic_carbon_target.csv and lucas_organic_carbon_training_and_test_data.csv have to be located in this directory for now
- to run this project use the python module streamlit like this: "streamlit run streamlit_app.py"

## Purpose and scope
- Milestone1
Task1: Identify Overall Misclassification Patterns
Solution: Confusion Matrix

Task2: Compare True Labels with Predicted Labels to Identify Major Misclassifications
Solution: Confusion Matrix, PCA Scatterplot

Task3: Detect Class Imbalance
Solution: Confusion Matrix, Classification Statistics Table that shows the count of True Positives, False positives and False Negatives

Task4: Explore classification error for specification groups of data
Solution: when clicking the confusion matrix a scatter plot with a subset according to selection is rendered

- Milestone2
...

## Components
Normalization option: done
Confustion Matrix: done
scatter plot: done
classification Statistics: done

## Major Implementation Activities
Normalization option: calculating normalized values and optimize internal logic for turning the option on and off
Confusion Matrix: picking fitting color scheme, implementing hover information
scatter plot: implementing PCA, link content to Confusion Matrix selection
classification Statistics: defining True Positives, False positives and False Negatives
