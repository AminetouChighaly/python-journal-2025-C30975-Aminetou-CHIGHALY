import folium
import pandas as pd
from folium.plugins import HeatMap

# =========================
# 1. DONNÉES
# =========================
centroids = pd.read_csv("cluster_centroids.csv")
metrics = pd.read_csv("fluidity_result.csv")

df = centroids.merge(metrics, on="cluster")

# =========================
# 2. CARTE CENTRÉE SUR CLUSTERS (IMPORTANT)
# =========================
m = folium.Map(
    location=[df["lat"].mean(), df["lon"].mean()],
    zoom_start=13
)

# =========================
# 3. CLUSTERS
# =========================
q1 = df["fluidity_score"].quantile(0.33)
q2 = df["fluidity_score"].quantile(0.66)

for _, row in df.iterrows():

    if row["fluidity_score"] > q2:
        color = "green"
    elif row["fluidity_score"] > q1:
        color = "orange"
    else:
        color = "red"

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=10,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"Cluster {row['cluster']} | Fluidité {row['fluidity_score']:.3f}"
    ).add_to(m)

# =========================
# 4. HEATMAP (IMPORTANT CORRECTION)
# =========================
points = pd.read_csv("simulation_output.csv")

heat_data = points[["lat", "lon"]].values.tolist()

HeatMap(
    heat_data,
    radius=8,
    blur=12
).add_to(m)

# =========================
# 5. BOUNDS (IMPORTANT FIX)
# =========================
all_lat = pd.concat([points["lat"], df["lat"]])
all_lon = pd.concat([points["lon"], df["lon"]])

m.fit_bounds([
    [all_lat.min(), all_lon.min()],
    [all_lat.max(), all_lon.max()]
])

# =========================
# 6. SAVE
# =========================
m.save("map.html")