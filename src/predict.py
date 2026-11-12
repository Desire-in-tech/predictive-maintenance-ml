"""Inference wrapper reused by Task 10."""
import joblib
import pandas as pd

class PredictiveMaintenanceModel:
    def __init__(self, artifact_path="artifacts/predictive_maintenance_pipeline.joblib"):
        a=joblib.load(artifact_path)
        self.pipeline=a["pipeline"]; self.threshold=float(a["threshold"]); self.features=list(a["features"])
    def predict(self, readings):
        frame=pd.DataFrame(readings)
        missing=[x for x in self.features if x not in frame]
        extra=[x for x in frame if x not in self.features]
        if missing or extra: raise ValueError(f"Feature mismatch: missing={missing}, unexpected={extra}")
        probs=self.pipeline.predict_proba(frame[self.features])[:,1]
        return [{"failure_probability":float(p),"predicted_failure":bool(p>=self.threshold),
                 "threshold":self.threshold} for p in probs]
