import pandas as pd
import numpy as np
import os

def load_user_data(user_id="001"):
    
    path = f"Geolife Trajectories 1.3/Data/{user_id}/Trajectory/"
    all_data = []

    # =========================
    # 1. CHARGEMENT
    # =========================
    for file in os.listdir(path):
        if file.endswith(".plt"):
            file_path = os.path.join(path, file)

            df = pd.read_csv(
                file_path,
                skiprows=6,
                header=None,
                usecols=[0, 1, 3, 5, 6],
                names=['lat', 'lon', 'alt', 'date', 'time']
            )

            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            df = df[['timestamp', 'lat', 'lon', 'alt']]

            all_data.append(df)

    #si la liste est vide
    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # =========================
    # 2. NETTOYAGE
    # =========================
    df = df.drop_duplicates()
    #Suppression des coordonnées invalides
    df = df[
        (df["lat"].between(-90, 90)) &
        (df["lon"].between(-180, 180))
    ]

    # =========================
    # 3. ANALYSE DE MOUVEMENT
    # =========================

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    df["distance_km"] = haversine(
        df["lat"].shift(),
        df["lon"].shift(),
        df["lat"],
        df["lon"]
    )

    df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 3600

    # éviter division par 0
    df["time_diff"] = df["time_diff"].replace(0, np.nan)

    df["speed_kmh"] = df["distance_km"] / df["time_diff"]
        

    # =========================
    # 4. COMPORTEMENT
    # =========================
    def behavior(speed):
        if pd.isna(speed):
            return "STOP"
        elif speed < 1:
            return "STOP"
        elif speed < 5:
            return "WALK"
        else:
            return "MOVE"

    df["behavior"] = df["speed_kmh"].apply(behavior)
    
    # =========================
    # utilisé pour FLUIDITÉ
    # =========================
    #corriger la vitesse en fonction du comportement pour éliminer le bruit
    def adjust_speed(row):
            if row["behavior"] == "STOP":
                return 0
            return row["speed_kmh"]
        
    df["speed_kmh"] = df.apply(adjust_speed, axis=1)
    
    #  supprimer valeurs absurdes
    df = df[(df["speed_kmh"] < 150) | (df["speed_kmh"].isna())]
    
    # =========================
    # 5. FINAL
    # =========================
    df["user_id"] = user_id

    return df


if __name__ == "__main__":
    df = load_user_data("001")

    print(df.head()) #affiche les 5 premières lignes du DataFrame
    print(df.tail()) #affiche les 5 dernières lignes
    print(df.columns) #affiche les noms des colonnes
    print(df.shape) #affiche la taille
    df.to_csv("cleaned_data.csv", index=False)