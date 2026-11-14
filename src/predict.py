"""Inference wrappers for Task 9 and the Task 10 handoff."""
import joblib
import pandas as pd


class PredictiveMaintenanceModel:
    def __init__(
        self,
        artifact_path="artifacts/predictive_maintenance_pipeline.joblib",
        failure_modes_artifact_path=None,
    ):
        artifact = joblib.load(artifact_path)
        self.pipeline = artifact["pipeline"]
        self.threshold = float(artifact["threshold"])
        self.features = list(artifact["features"])
        self.failure_modes_artifact = (
            joblib.load(failure_modes_artifact_path)
            if failure_modes_artifact_path
            else None
        )

    def _validate_readings(self, readings):
        frame = pd.DataFrame(readings)
        missing = [column for column in self.features if column not in frame]
        extra = [column for column in frame if column not in self.features]
        if missing or extra:
            raise ValueError(
                f"Feature mismatch: missing={missing}, unexpected={extra}"
            )
        return frame[self.features]

    def predict(self, readings):
        """Return the original binary prediction response for compatibility."""
        frame = self._validate_readings(readings)
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        return [
            {
                "failure_probability": float(probability),
                "predicted_failure": bool(probability >= self.threshold),
                "threshold": self.threshold,
            }
            for probability in probabilities
        ]

    def predict_with_failure_modes(self, readings):
        """Return overall failure predictions plus independent mode predictions."""
        if self.failure_modes_artifact is None:
            raise RuntimeError(
                "Load failure_modes_pipeline.joblib using "
                "failure_modes_artifact_path to enable mode predictions."
            )

        frame = self._validate_readings(readings)
        overall = self.predict(readings)
        artifact = self.failure_modes_artifact
        results = []

        for index, base_prediction in enumerate(overall):
            mode_predictions = {}
            predicted_modes = []
            for mode in artifact["failure_modes"]:
                probability = float(
                    artifact["models"][mode].predict_proba(frame.iloc[[index]])[0, 1]
                )
                threshold = float(artifact["thresholds"][mode])
                predicted = bool(probability >= threshold)
                mode_predictions[mode] = {
                    "probability": probability,
                    "predicted": predicted,
                    "threshold": threshold,
                }
                if predicted:
                    predicted_modes.append(mode)

            results.append(
                {
                    **base_prediction,
                    "failure_modes": mode_predictions,
                    "predicted_failure_modes": predicted_modes,
                }
            )
        return results
