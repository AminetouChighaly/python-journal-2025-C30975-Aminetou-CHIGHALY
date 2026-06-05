import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

# =========================
# 1. CHARGER DONNÉES
# =========================
df = pd.read_csv("simulation_output.csv")

# garder uniquement GPS
coords = df[["lat", "lon"]].values

# =========================
# 2. DBSCAN
# =========================
# eps = distance (à ajuster)
# min_samples = densité minimale

db = DBSCAN(eps=0.5 / 6371 , min_samples=8, metric="haversine")#eps=100m

# convertir en radians (IMPORTANT pour GPS)
coords_rad = np.radians(coords)

df["cluster"] = db.fit_predict(coords_rad)
#donner un eposiiton unique pour une zone
centroids = df[df["cluster"] != -1].groupby("cluster").agg({
    "lat": "mean",
    "lon": "mean"
}).reset_index()

centroids.to_csv("cluster_centroids.csv", index=False)

# =========================
# 3. ANALYSE DES CLUSTERS
# =========================
print(" Nombre de clusters :", len(set(df["cluster"])) - (1 if -1 in df["cluster"] else 0))
print("Bruit :", sum(df["cluster"] == -1))

# =========================
# 4. TOP HOTSPOTS
# =========================
hotspots = df[df["cluster"] != -1].groupby("cluster").agg({
    "lat": "mean",
    "lon": "mean",
    "user_id": "count"
}).rename(columns={"user_id": "size"}).sort_values("size", ascending=False)

print("\n Hotspots détectés :")
print(hotspots.head(10))

# =========================
# 5. SAUVEGARDE
# =========================
df.to_csv("clustered_output.csv", index=False)
hotspots.to_csv("hotspots.csv")

print("\n Fichiers générés : clustered_output.csv + hotspots.csv")