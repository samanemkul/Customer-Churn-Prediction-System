import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "churn_model.joblib"

model = joblib.load(MODEL_PATH)


def predict_churn(customer_data):
    probability = model.predict_proba(customer_data)[0][1]

    prediction = int(probability >= 0.45)

    return {
        "prediction": prediction,
        "churn_probability": float(probability),
        "risk_level": "High" if prediction == 1 else "Low"
    }