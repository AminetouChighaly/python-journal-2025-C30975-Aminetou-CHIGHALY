# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 17:55:08 2025

@author: Republic Of Computer
"""

# Import des librairies
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Télécharger les stopwords français
nltk.download("stopwords")

# Exemple de dataset (avis simples en français)
documents = [
    "J'adore ce produit, il est incroyable !",
    "C'est le pire achat que j'ai jamais fait.",
    "Absolument fantastique, je suis très satisfait.",
    "Je déteste cet article, très décevant.",
    "Bonne qualité et livraison rapide.",
    "Horrible, une perte d'argent."
]

# Prétraitement avec TF-IDF et stopwords français
vectorizer = TfidfVectorizer(stop_words=stopwords.words('french'))
X = vectorizer.fit_transform(documents)

# K-Means avec 2 clusters (positif/négatif attendus)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

# Résultats
print("Clusters attribués :", kmeans.labels_)

# Afficher les documents avec leur cluster
for i, doc in enumerate(documents):
    print(f"Cluster {kmeans.labels_[i]} --> {doc}")