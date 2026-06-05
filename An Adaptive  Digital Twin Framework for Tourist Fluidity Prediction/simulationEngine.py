from data_process import load_user_data
from digital_twin import TouristTwin
import pandas as pd
import time
import numpy as np
# =========================
# FEEDBACK SYSTEM INIT
# =========================
#from feedback_engine import FeedbackEngine

# engine = FeedbackEngine()
# engine.build_rules()
# =========================
# 1. PARAMÈTRES
# =========================
NUM_USERS = 10
MAX_STEPS = 1000

# =========================
# 2. CHARGEMENT DES AGENTS
# =========================
agents = []

for i in range(1, NUM_USERS + 1):
    user_id = str(i).zfill(3)
    df_user = load_user_data(user_id)

    if df_user.empty: #si les données de l'user sont vides on passe au suivant
        continue

    agents.append(TouristTwin(user_id, df_user))


print(f"{len(agents)} agents chargés")

# =========================
# 3. SIMULATION ENGINE
# =========================
global_history = []

for step in range(MAX_STEPS):

    active_agents = 0

    for agent in agents:

        result = agent.step()

        # arrêt agent
        if result is None or "finished" in result:
            continue
        # =========================
        # FEEDBACK LOOP 
        # =========================
        #cluster = result.get("cluster", -1)
        # rule = engine.get_rule(cluster)

        #adjusted_speed = result["speed_kmh"] * rule["speed_factor"]
        #agent.current_speed = adjusted_speed
        # =========================
        # NORMALISATION DONNÉES
        # =========================
        global_history.append({
           "user_id": result["user_id"],
           "lat": result["lat"],
           "lon": result["lon"],
            "behavior": result["behavior"],
            "timestamp": result["timestamp"],
            "speed_kmh": result.get("speed_kmh", None)
        })
        

    # =========================
    # FEEDBACK
    # =========================
    # global_history.append({
    #         "user_id": result["user_id"],
    #         "lat": result["lat"],
    #         "lon": result["lon"],
    #         "behavior": result["behavior"],
    #         "timestamp": result["timestamp"],
    #         "speed_kmh": result.get("speed_kmh", None),
    #         "cluster": cluster,
    #         "adjusted_speed": adjusted_speed,
    #         "feedback_action": rule.get("action", "NEUTRAL")
    #     })

        active_agents += 1

    if active_agents == 0:
        print("\n Simulation terminée (tous les agents ont fini)")
        break

    

# =========================
# 4. RESULTATS
# =========================
print("\n Simulation terminée")
print(f"Total points générés: {len(global_history)}")

# =========================
# 5. SAUVEGARDE
# =========================
df = pd.DataFrame(global_history)
df.to_csv("simulation_output.csv", index=False)

print("Fichier sauvegardé : simulation_output.csv")