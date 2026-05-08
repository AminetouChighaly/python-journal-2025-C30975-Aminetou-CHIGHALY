import pandas as pd
import folium

# =========================
# CHARGER DONNÉES
# =========================

df = pd.read_csv("simulation_output.csv")

# =========================
# CRÉER CARTE
# =========================

m = folium.Map(
    location=[df["lat"].mean(), df["lon"].mean()],
    zoom_start=12
)

# =========================
# COULEURS FEEDBACK
# =========================

def get_color(action):

    if action == "ATTRACT":
        return "green"

    elif action == "AVOID":
        return "red"

    else:
        return "orange"

# =========================
# AJOUT POINTS
# =========================

for _, row in df.iterrows():

    color = get_color(row["feedback_action"])

    popup_text = (
        f"Cluster: {row['cluster']}<br>"
        f"Action: {row['feedback_action']}<br>"
        f"Speed: {row['adjusted_speed']:.2f}"
    )

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=popup_text
    ).add_to(m)

# =========================
# SAUVEGARDE
# =========================

m.save("feedback_map.html")

print("Carte feedback générée : feedback_map.html")

