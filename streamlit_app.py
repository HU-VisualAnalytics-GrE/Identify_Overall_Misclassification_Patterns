# -------------- Imports --------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from streamlit_plotly_events import plotly_events
from sklearn.decomposition import PCA

# -------------- Constants --------------
CLASS_MAPPING = {
    0: 'very_high',
    1: 'high',
    2: 'moderate',
    3: 'low',
    4: 'very_low'
}

# -------------- Helper Functions --------------
def get_label_from_index(index):
    return CLASS_MAPPING[index]

# -------------- Data Loading and Preparation --------------
def load_and_prepare_data():
    df_target = pd.read_csv("lucas_organic_carbon_target.csv")
    df_test = pd.read_csv("lucas_organic_carbon_training_and_test_data.csv")
    
    X = df_test  # Features
    y = df_target['x']  # Labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=42)
    return X_train, X_test, y_train, y_test

# -------------- Model Training --------------
def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    return rf_model

# -------------- Confusion Matrix Functions --------------
def analyze_misclassifications(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    st.title("Classification Errors")
    return cm, cmn

def confussion_matrix(cm):
    class_names = ['very high', 'high', 'moderate', 'low', 'very low']
    fig = px.imshow(cm, text_auto='.2f', color_continuous_scale="blues")
    fig.update_layout(
        xaxis_title="Tatsächliche Klasse",
        yaxis_title="Vorhergesagte Klasse",
        xaxis=dict(ticktext=class_names, tickvals=list(range(len(class_names)))),
        yaxis=dict(ticktext=class_names, tickvals=list(range(len(class_names))))
    )
    fig.update_traces(
        hovertemplate='<b>Vorhergesagte Klasse: %{y}</b><br>Tatsächliche Klasse: %{x}<br>Wert: %{z}<extra></extra>',
        hoverlabel=dict(bgcolor="grey", font_size=16, font_family="Rockwell"),
        showscale=True,
        colorscale="Blues",
        colorbar=dict(title="Normierte Häufigkeit"),
    )
    
    #st.plotly_chart(fig)
    selected_points = plotly_events(fig)
    
    if selected_points:
        point = selected_points[0]
        true_label = get_label_from_index(int(point['x']))
        pred_label = get_label_from_index(int(point['y']))
        show_pca_for_labels(true_label, pred_label)

def confussion_matrix_normalized(cmn):
    class_names = ['very high', 'high', 'medium', 'low', 'very low']
    fig = px.imshow(cmn, text_auto='.2f', color_continuous_scale="blues")
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        legend_title="Classification",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100),
        xaxis_title="Tatsächliche Klasse",
        yaxis_title="Vorhergesagte Klasse",
        xaxis=dict(ticktext=class_names, tickvals=list(range(len(class_names)))),
        yaxis=dict(ticktext=class_names, tickvals=list(range(len(class_names))))
    )
    
    fig.update_traces(
        hovertemplate='<b>Vorhergesagte Klasse: %{y}</b><br>Tatsächliche Klasse: %{x}<br>Wert: %{z}<extra></extra>',
        hoverlabel=dict(bgcolor="lightgrey", font_size=16, font_family="Rockwell"),
        showscale=True,
        colorscale="Blues",
        colorbar=dict(title="Normierte Häufigkeit"),
    )
    
    #st.plotly_chart(fig)
    selected_points = plotly_events(fig)
    
    if selected_points:
        point = selected_points[0]
        true_label = get_label_from_index(int(point['x']))
        pred_label = get_label_from_index(int(point['y']))
        show_pca_for_labels(true_label, pred_label)

# -------------- PCA Functions --------------
def prepare_pca_data(X_test, y_test, predictions):
    pca = PCA(n_components=2, random_state=42)
    pca_features = pca.fit_transform(X_test)

    return pd.DataFrame({
        'PC1': pca_features[:, 0],
        'PC2': pca_features[:, 1],
        'TrueLabel': y_test,
        'PredictedLabel': predictions
    })

def adjust_df_for_plot(df, true_label, predicted_label):
    filtered_df = df[(df['TrueLabel'] == true_label) | (df['PredictedLabel'] == predicted_label)].copy()
    filtered_df['classification'] = np.select(
            condlist=[
                (filtered_df['TrueLabel'] == true_label) & (filtered_df['PredictedLabel'] == predicted_label),
                (filtered_df['TrueLabel'] == true_label) & (filtered_df['PredictedLabel'] != predicted_label),
                (filtered_df['TrueLabel'] != true_label) & (filtered_df['PredictedLabel'] == predicted_label),
            ],
            choicelist=['True Positives', 'False Negatives', 'False Positives'],
            default='Other'
        )
    return filtered_df

def scatter_plot_df(df, true_label, predicted_label):
    color_map = {
        'True Positives': '#00FF00',
        'False Negatives': '#FF0000',
        'False Positives': '#0000FF'
    }
    
    hover_data = {
        'PC1': True,
        'PC2': True,
        'TrueLabel': True,
        'PredictedLabel': True,
        'classification': True
    }
    
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='classification',
        color_discrete_map=color_map,
        hover_data=hover_data,
        title=f'PCA Visualization for True: {true_label}, Predicted: {predicted_label}'
    )
    
    fig.update_layout(
        legend_title="Classification",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100)
    )
    
    return fig

def show_pca_for_labels(true_label, pred_label):
    pca_df = prepare_pca_data(X_test, y_test, predictions)
    filtered_df = adjust_df_for_plot(pca_df, true_label, pred_label)
    fig = scatter_plot_df(filtered_df, true_label, pred_label)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Classification Statistics")
    stats = filtered_df['classification'].value_counts()
    st.write(pd.DataFrame({
        'Classification': stats.index,
        'Count': stats.values
    }))

# -------------- Main Execution --------------
if __name__ == "__main__":
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Train model and make predictions
    rf_model = train_model(X_train, y_train)
    predictions = rf_model.predict(X_test)
    
    # Create confusion matrix
    cm, cmn = analyze_misclassifications(y_test, predictions)
    
    # Setup Streamlit interface
    st.sidebar.title("Options")
    option_cmn = st.sidebar.checkbox("Normalize", value=True)
    
    # Display appropriate matrix
    if option_cmn:
        confussion_matrix_normalized(cmn)
    else:
        confussion_matrix(cm)