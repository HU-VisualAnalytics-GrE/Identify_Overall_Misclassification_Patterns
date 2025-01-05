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




df_target = pd.read_csv("lucas_organic_carbon_target.csv")
df_test = pd.read_csv("lucas_organic_carbon_training_and_test_data.csv")

# Daten vorbereiten
X = df_test  # Features
y = df_target['x']  # Labels/Zielvariable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=42)

# Random Forest Modell erstellen und trainieren
rf_model = RandomForestClassifier(
    n_estimators=100,      # Anzahl der Bäume
    max_depth=None,        # Maximale Tiefe der Bäume
    random_state=42,       # Für Reproduzierbarkeit
    n_jobs=-1              # Nutzt alle verfügbaren CPU-Kerne
)

# Modell trainieren
rf_model.fit(X_train, y_train)

# Vorhersagen machen
predictions = rf_model.predict(X_test)

# Modell evaluieren
#print("Accuracy:")
#print(accuracy_score(y_test, predictions))

#print("Klassifikationsbericht:")
#print(classification_report(y_test, predictions))

def analyze_misclassifications(y_test, predictions):
    # Generiere Konfusionsmatrix
    cm = confusion_matrix(y_test, predictions)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    st.title("Missklassifizierungen")
    return cm,cmn
def confussion_matrix(cm):
    class_names = ['very high', 'high', 'medium', 'low', 'very low']
    fig = px.imshow(cm, text_auto='.2f', color_continuous_scale="blues")
    fig.update_layout(
        xaxis_title="Tatsächliche Klasse",
        yaxis_title="Vorhergesagte Klasse",
        xaxis=dict(ticktext=class_names, tickvals=list(range(len(class_names)))),
        yaxis=dict(ticktext=class_names, tickvals=list(range(len(class_names))))
    )

    fig.update_traces(
        hovertemplate=
        '<b>Vorhergesagte Klasse: %{y}</b><br>' +
        'Tatsächliche Klasse: %{x}<br>' +
        'Wert: %{z}<extra></extra>',  # Anzeige der Zelle, auf die der Benutzer zeigt
        hoverlabel=dict(
            bgcolor="grey",  # Hintergrundfarbe der Hover-Box
            font_size=16,      # Schriftgröße
            font_family="Rockwell"  # Schriftart der Hover-Box
        ),
        showscale=True,  # Farbschattierung anzeigen
        colorscale="Blues",  # Festgelegte Farben für das Diagramm
        colorbar=dict(title="Normierte Häufigkeit"),  # Farbbalken einfügen
    )

    st.plotly_chart(fig)
    #selected_points = plotly_events(fig)
    #st.write(selected_points)
    return cm#, selected_points

def confussion_matrix_normalized(cmn):
    # Normalisierte Heatmap
    class_names = ['very high', 'high', 'medium', 'low', 'very low']
    fig = px.imshow(cmn, text_auto='.2f', color_continuous_scale="blues")
    fig.update_layout(
        xaxis_title="Tatsächliche Klasse",
        yaxis_title="Vorhergesagte Klasse",
        xaxis=dict(ticktext=class_names, tickvals=list(range(len(class_names)))),
        yaxis=dict(ticktext=class_names, tickvals=list(range(len(class_names))))
    )
    fig.update_traces(
        hovertemplate=
        '<b>Vorhergesagte Klasse: %{y}</b><br>' +
        'Tatsächliche Klasse: %{x}<br>' +
        'Wert: %{z}<extra></extra>',  # Anzeige der Zelle, auf die der Benutzer zeigt
        hoverlabel=dict(
            bgcolor="grey",  # Hintergrundfarbe der Hover-Box
            font_size=16,      # Schriftgröße
            font_family="Rockwell"  # Schriftart der Hover-Box
        ),
        showscale=True,  # Farbschattierung anzeigen
        colorscale="Blues",  # Festgelegte Farben für das Diagramm
        colorbar=dict(title="Normierte Häufigkeit"),  # Farbbalken einfügen
    )
    st.plotly_chart(fig)
    #selected_points = plotly_events(fig)
    #st.write(selected_points)
    return cmn#,selected_points


cm,cmn = analyze_misclassifications(y_test, predictions)
st.sidebar.title("Options")
option_cmn = st.sidebar.checkbox("Normalize", value=True)

if option_cmn:
    confussion_matrix_normalized(cmn)
else:
    confussion_matrix(cm)
