class DecisionEngine:

    def compute_health(self, training, drift, profile):
        best = (training.get("leaderboard") or [{}])[0]

        acc = best.get("metrics", {}).get("accuracy") or best.get("primary_metric", 0)
        drift_score = drift.get("drift_score", 0)
        missing = profile.get("missing_pct", 0)

        health = (
            acc * 0.5 +
            (1 - min(drift_score, 0.3)) * 0.3 +
            (1 - min(missing, 0.3)) * 0.2
        ) * 100

        status = "Healthy"
        if health < 80:
            status = "Warning"
        if health < 60:
            status = "Critical"

        return round(health, 2), status

    def analyze(self, session_data):
        training = session_data.get("training_results", {})
        drift = session_data.get("drift", {})
        profile = session_data.get("profile", {})

        best = (training.get("leaderboard") or [{}])[0]
        acc = best.get("primary_metric", 0)
        drift_score = drift.get("drift_score", 0)

        actions = []

        if acc < 0.8:
            actions.append({
                "type": "warning",
                "title": "Low Accuracy",
                "description": f"{round(acc,2)}",
                "action": "optimize_model"
            })

        if drift_score > 0.1:
            actions.append({
                "type": "danger",
                "title": "Data Drift",
                "description": f"{round(drift_score,3)}",
                "action": "retrain_model"
            })

        health, status = self.compute_health(training, drift, profile)

        return {
            "actions": actions,
            "health_score": health,
            "health_status": status
        }