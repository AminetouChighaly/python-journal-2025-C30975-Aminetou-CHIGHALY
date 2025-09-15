# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 15:01:14 2025

@author: Republic Of Computer
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Charger le dataset
df = pd.read_csv("dataset_sentiment_reseaux.csv")

print("Aperçu du dataset :")
print(df.head())

# 2. Définir les variables explicatives (features) et la cible (target)
X = df[["Nb_Hashtags", "Nb_Mots", "Nb_Likes"]]   # variables indépendantes
y = df["Sentiment_Score"]                        # variable cible

# 3. Créer et entraîner le modèle
model = LinearRegression()
model.fit(X, y)

# 4. Afficher les coefficients
print("\nCoefficients du modèle :")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col}: {coef:.4f}")

print(f"Intercept: {model.intercept_:.4f}")

# 5. Prédictions
df["Prediction"] = model.predict(X)
print("\nPrédictions comparées aux vraies valeurs :")
print(df[["Sentiment_Score", "Prediction"]])

# 6. Visualisation (exemple avec Nb_Likes)
plt.scatter(df["Nb_Likes"], y, color="blue", label="Vrai score")
plt.scatter(df["Nb_Likes"], df["Prediction"], color="red", label="Prédiction")
plt.xlabel("Nb_Likes")
plt.ylabel("Sentiment_Score")
plt.legend()
plt.title("Régression linéaire : Score de sentiment en fonction des Likes")
plt.show()