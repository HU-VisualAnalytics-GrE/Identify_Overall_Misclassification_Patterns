import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


def train_model_and_save_data():
    df_target = pd.read_csv("lucas_organic_carbon_target.csv")
    df_test = pd.read_csv("lucas_organic_carbon_training_and_test_data.csv")

    # prepare data
    X = df_test  # Features
    y = df_target['x']  # Labels

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=42)

    # initialize model
    rf_model = RandomForestClassifier(
        n_estimators=100,      # Number of estimators
        max_depth=None,        # max depth of the tree
        random_state=42,       # for reproducibility
        n_jobs=-1              # uses all core threads
    )

    # train model
    rf_model.fit(X_train, y_train)

    # make predictions
    predictions = rf_model.predict(X_test)

    # save orig labels, predicted labels and features into a file for faster usage
    y_test.to_csv("true_test_labels.csv", index=False, header=False)
    np.savetxt("predicted_labels.csv", predictions, delimiter="\n", fmt="%s")
    X_test.to_csv("test_features.csv", index=False)

    # evaluate model
    print("Accuracy:")
    print(accuracy_score(y_test, predictions))

    print("Classification Report:")
    print(classification_report(y_test, predictions))

    pd.set_option('display.float_format', '{:.16f}'.format)
    features = pd.read_csv('test_features.csv')
    pca = PCA(n_components=2, random_state=42)
    low_features_pca = pca.fit_transform(features)
    np.savetxt("pca_2_features.csv", low_features_pca, delimiter=",")


def load_df():
    true_labels_df = pd.read_csv('true_test_labels.csv', header=None, names=['TrueLabel'])
    predicted_labels_df = pd.read_csv('predicted_labels.csv', header=None, names=['PredictedLabel'])
    pca_features_df = pd.read_csv('pca_2_features.csv', header=None, names=['PC1', 'PC2'])
    df = pd.concat([true_labels_df, predicted_labels_df, pca_features_df], axis=1)
    return df


def adjust_df_for_plot(df, true_label="very_low", predicted_label="very_low"):
    filtered_df = df[(df['TrueLabel'] == true_label) | (df['PredictedLabel'] == predicted_label)].copy()
    filtered_df['classification'] = np.select(
            condlist=[
                (filtered_df['TrueLabel'] == true_label) & (filtered_df['PredictedLabel'] == predicted_label),  # True Positive
                (filtered_df['TrueLabel'] == true_label) & (filtered_df['PredictedLabel'] != predicted_label),  # False Negative
                (filtered_df['TrueLabel'] != true_label) & (filtered_df['PredictedLabel'] == predicted_label),  # False Positive
            ],
            choicelist=['TP', 'FN', 'FP'],
            default='Other'
        )

    color_map = {'TP': 'green', 'FN': 'red', 'FP': 'blue'}
    filtered_df['color'] = filtered_df['classification'].map(color_map)

    return filtered_df


def scatter_plot_df(df, true_label="very_low", predicted_label="very_low"):

    plt.figure(figsize=(10, 7))
    plt.scatter(df['PC1'], df['PC2'], c=df['color'])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Scatter Plot for True: {true_label}, Predicted: {predicted_label}')

    # dynamic legend
    legend_labels = {
        'TP': mpatches.Patch(color='green', label=f'True: {true_label}, Predicted: {predicted_label} (True Positives)'),
        'FN': mpatches.Patch(color='red', label=f'True: {true_label}, Predicted: Other (False Negatives'),
        'FP': mpatches.Patch(color='blue', label=f'True: Other, Predicted: {predicted_label} (False Positives)')
    }

    # move legend out of scatter plot
    plt.legend(
        handles=legend_labels.values(),
        loc='upper center',           # Place the legend above the anchor point
        bbox_to_anchor=(0.5, -0.1),  # Anchor at the center below the plot
        ncol=3                       # Display the legend in a single row
    )
    plt.subplots_adjust(bottom=0.15)
    plt.show()


if __name__ == '__main__':
    df = load_df()
    df = adjust_df_for_plot(df)
    scatter_plot_df(df)
