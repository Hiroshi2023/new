import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """Charge les données depuis le fichier CSV"""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Prétraitement des données"""
    # Suppression de la colonne Unnamed
    df.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Encodage des variables catégorielles
    le = LabelEncoder()
    df['cut'] = le.fit_transform(df['cut'])
    
    # Encodage frequency encoding pour color et clarity
    fe_color = df.groupby('color').size()/len(df)
    df['color_encoded'] = df['color'].map(fe_color)
    
    fe_clarity = df.groupby('clarity').size()/len(df)
    df['clarity_encoded'] = df['clarity'].map(fe_clarity)
    
    # Suppression des colonnes originales
    df.drop(['color', 'clarity'], axis=1, inplace=True)
    
    return df

def analyze_data(df):
    """Analyse exploratoire des données"""
    # Matrice de corrélation
    cat_col = ['cut']
    num_df = df.drop(cat_col, axis=1)
    sns.heatmap(num_df.corr(), annot=True)
    plt.show()
    
    return df.describe()

import os
from script.data_processing import load_data, preprocess_data, analyze_data

if __name__ == "__main__":
    # Obtenir le chemin absolu du fichier actuel (ex: script/mon_script.py)
    current_dir = os.path.dirname(__file__)

    # Remonter au dossier parent (ex: projet/), puis aller dans data/diamonds.csv
    data_path = os.path.abspath(os.path.join(current_dir, "..", "data", "diamonds.csv"))

    # Chargement et traitement des données
    data = load_data(data_path)
    data = preprocess_data(data)
    stats = analyze_data(data)
