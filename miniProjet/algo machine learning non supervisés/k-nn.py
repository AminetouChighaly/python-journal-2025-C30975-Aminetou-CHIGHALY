# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 14:29:05 2025

@author: Republic Of Computer
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Chargement (ajustez le chemin et les colonnes si nécessaire)
COLUMNS = ['sentiment', 'id', 'date', 'query', 'user', 'text']
df = pd.read_csv(
    '../training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    header=None,
    names=COLUMNS
)
df = df[['sentiment', 'text']]
df['sentiment'] = df['sentiment'].replace(4, 1)

# 2. Fonction de Nettoyage (simplifiée pour l'exemple)
def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

df['text'] = df['text'].apply(clean_tweet)
# Séparer les données
X = df['text']
y = df['sentiment']
# Utiliser un sous-échantillon pour k-NN car il est lent sur 1.6M de lignes
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.01, random_state=42) # Utilise 1% pour le test, 99% pour le train

# Réduire la taille de l'ensemble d'entraînement pour des raisons de performance (k-NN est lent)
# Nous allons prendre 50 000 échantillons au hasard pour l'entraînement
X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, train_size=50000, random_state=42, stratify=y_train_full)
X_test = X_test_full
y_test = y_test_full

# Créer le vectoriseur TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# Transformer les ensembles
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Créer l'instance du modèle k-NN
# n_neighbors est le 'k'. Nous utilisons 'n_jobs=-1' pour utiliser tous les coeurs du processeur.
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

# 2. Entraînement du modèle
# L'entraînement est rapide car le modèle ne fait que stocker les données.
print("Début de l'entraînement du modèle k-NN (stockage des données)...")
knn_model.fit(X_train_vec, y_train)
print("Entraînement terminé.")

# 3. Prédiction
# La prédiction est la partie lente, car elle nécessite le calcul des distances.
print("Début de la prédiction (calcul des distances)...")
y_pred_knn = knn_model.predict(X_test_vec)
print("Prédiction terminée.")
# Afficher les métriques
print("\n--- Évaluation du Modèle k-NN (k=5) ---")
print(f"Exactitude (Accuracy): {accuracy_score(y_test, y_pred_knn):.4f}")
print("\nRapport de Classification:")
print(classification_report(y_test, y_pred_knn))