# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 15:11:53 2025

@author: Republic Of Computer
"""

import pandas as pd

# Données d'exemple
data = {
    'critique': [
        "Ce film est fantastique, j'ai adoré chaque instant.",
        "Quelle perte de temps. Je me suis ennuyé à mourir.",
        "Un chef-d'œuvre, à voir absolument.",
        "Très décevant, l'histoire était faible.",
        "Un film génial et divertissant.",
        "Je ne recommanderais pas ce film à quiconque."
    ],
    'sentiment': [1, 0, 1, 0, 1, 0]  # 1 pour positif, 0 pour négatif
}
df = pd.DataFrame(data)
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialisation du vectoriseur TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=100) # On retire les mots courants

# Transformation des critiques en vecteurs numériques
X = vectorizer.fit_transform(df['critique'])
y = df['sentiment']
from sklearn.linear_model import LogisticRegression

# Initialisation du modèle
model = LogisticRegression()

# Entraînement du modèle sur nos données
model.fit(X, y)
# Nouvelles critiques à analyser
new_critics = [
    "C'était une expérience incroyable, j'ai beaucoup aimé.",
    "Un film horrible et sans intérêt."
]

# Il faut vectoriser les nouvelles critiques de la même manière que les données d'entraînement
X_new = vectorizer.transform(new_critics)

# Prédiction
predictions = model.predict(X_new)

# Affichage des résultats
for critique, sentiment in zip(new_critics, predictions):
    print(f"Critique : '{critique}' -> Sentiment Prédit : {'Positif' if sentiment == 1 else 'Négatif'}")

# Sortie attendue :
# Critique : 'C'était une expérience incroyable, j'ai beaucoup aimé.' -> Sentiment Prédit : Positif
# Critique : 'Un film horrible et sans intérêt.' -> Sentiment Prédit : Négatif