import pandas as pd
import numpy as np
# =========================
# 1. CHARGER LES DONNÉES CLUSTERISÉES
# =========================
df = pd.read_csv("clustered_output.csv")
print(df.columns)
# =========================
# 2. SUPPRIMER LE BRUIT (-1 si existant)
# =========================
df = df[df["cluster"] != -1]

# =========================
# 3. CALCUL DES INDICATEURS PAR CLUSTER
# =========================

# vitesse moyenne
speed_mean = df.groupby("cluster")["speed_kmh"].mean()

# densité (nombre de points par cluster)
density = df.groupby("cluster").size()

# STOP ratio
#compte combien de STOP dans chaque cluster
stop_counts = df[df["behavior"] == "STOP"].groupby("cluster").size()
#forcer tous les clusters à exister 
stop_ratio = stop_counts.reindex(density.index, fill_value=0) / density 
#fill_value remplacer les valeurs manquantes créées lors d’un alignement par 0

# =========================
# 4. SCORE DE FLUIDITÉ
# =========================
fluidity = speed_mean / (1 + np.log1p(density) * stop_ratio)

# =========================
# 5. TABLE FINALE
# =========================
result = pd.concat(
    [speed_mean, density, stop_ratio, fluidity],
    axis=1 #axis indique dans quelle direction(h v) tu appliques une opération.
)

result.columns = [
    "speed_mean",
    "density",
    "stop_ratio",
    "fluidity_score"
]
# classement (important)
result = result.sort_values("fluidity_score", ascending=False)

# =========================
# 6. AFFICHAGE
# =========================
print("\n ANALYSE DE FLUIDITÉ TOURISTIQUE\n")
print(result.round(3)) #arrondit les valeurs numériques à 3 chiffres après la virgule

# =========================
# 7. SAUVEGARDE
# =========================
result.to_csv("fluidity_result.csv")

print("\n Résultat sauvegardé : fluidity_result.csv")