import pandas as pd
import numpy as np

class FeedbackEngine:
    def __init__(self, fluidity_file="fluidity_result.csv"):
        self.df = pd.read_csv(fluidity_file)

        # seuils dynamiques
        self.q1 = self.df["fluidity_score"].quantile(0.33)
        self.q2 = self.df["fluidity_score"].quantile(0.66)

        self.rules = {}

    def build_rules(self):
        """
        Crée des règles par cluster
        """

        for _, row in self.df.iterrows():

            cluster_id = row["cluster"]
            score = row["fluidity_score"]

            if score < self.q1:
                action = "AVOID"      # congestion
                speed_factor = 0.5

            elif score < self.q2:
                action = "NEUTRAL"
                speed_factor = 1.0

            else:
                action = "ATTRACT"   # zone fluide
                speed_factor = 1.2

            self.rules[cluster_id] = {
                "action": action,
                "speed_factor": speed_factor,
                "fluidity": float(score)
            }

        return self.rules

    def get_rule(self, cluster_id):
        """
        Retourne la règle d'un cluster
        """
        return self.rules.get(cluster_id, {
            "action": "NEUTRAL",
            "speed_factor": 1.0,
            "fluidity": None
        })


if __name__ == "__main__":
    engine = FeedbackEngine()
    rules = engine.build_rules()

    print("\n FEEDBACK RULES\n")
    for k, v in rules.items():
        print(f"Cluster {k}: {v}")
