from feedback_engine import FeedbackEngine
import pandas as pd
class TouristTwin:
    def __init__(self, user_id, df):
        self.id = user_id
        self.df = df.reset_index(drop=True)
        
        self.index = 0
        self.current_position = None
        self.current_state = None
        self.current_time = None
        self.current_speed = None

        self.history = []
        #feedback
#         self.feedback_engine = FeedbackEngine()
#         self.feedback_engine.build_rules()
#         clusters_df = pd.read_csv("clustered_output.csv")
#         self.df = self.df.merge(
#             clusters_df[["lat", "lon", "cluster"]],
#             on=["lat", "lon"],
#             how="left"
# )
        
    def step(self):
        """
        Simulation pas à pas du touriste
        """

        # =========================
        # FIN DE DONNÉES
        # =========================
        #si on a tout parcouru  arrête la simulation du touriste
        if self.index >= len(self.df):
            return {
                "user_id": self.id,
                "finished": True
            }

        row = self.df.iloc[self.index] #récupérer une ligne précise du dataset à un instant donné de la simulation.

        # =========================
        # EXTRACTION DONNÉES
        # =========================
        lat = float(row["lat"])
        lon = float(row["lon"])
        behavior = row["behavior"]
        timestamp = row["timestamp"]
        speed = row.get("speed_kmh", None)
        #feedback 
        
       # cluster = row.get("cluster", -1)
        # =========================
        # MISE À JOUR ÉTAT
        # =========================
        self.current_position = (lat, lon)
        self.current_state = behavior
        self.current_time = timestamp
        self.current_speed = speed
        #feedback
        # rule = self.feedback_engine.get_rule(cluster)

        # if self.current_speed is not None:
        #     self.current_speed *= rule["speed_factor"]
        # =========================
        # HISTORIQUE COMPLET
        # =========================
        self.history.append({
            "lat": lat,
            "lon": lon,
            "behavior": behavior,
            "timestamp": timestamp,
            "speed_kmh": speed,
            #feedback
            # "cluster": cluster,
            # "feedback_action": rule["action"],
            # "adjusted_speed": self.current_speed
        })

        self.index += 1
        
        # =========================
        # OUTPUT
        # =========================
        return {
            "user_id": self.id,
            "lat": lat,
            "lon": lon,
            "behavior": behavior,
            "timestamp": timestamp,
            "speed_kmh": speed,  # output pour la simulation
            #feedback
            # "cluster": cluster,
            # "feedback_action": rule["action"],
            #"adjusted_speed": self.current_speed
        }