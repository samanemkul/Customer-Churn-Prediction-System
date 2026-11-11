import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "churn_model.joblib"

# Load model once when the API starts
model = joblib.load(MODEL_PATH)

THRESHOLD = 0.45

MODEL_FEATURE_MAPPING = {
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen",
    "tenure": "Tenure Months",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges",
}


def predict_churn(customer_data):

    # Rename API columns to match model training columns
    customer_data = customer_data.rename(
        columns=MODEL_FEATURE_MAPPING
    )

    # Debugging information
    print("\n--- Incoming Model Data ---")
    print(customer_data.dtypes)

    print("\n--- Incoming Values ---")
    print(customer_data.iloc[0].to_dict())

    # Get the positive class index
    classes = list(model.classes_)
    positive_class_index = classes.index(1)

    # Generate prediction probability
    probability = model.predict_proba(customer_data)[0][positive_class_index]

    prediction = int(probability >= THRESHOLD)

    return {
        "prediction": prediction,
        "churn_probability": float(probability),
        "risk_level": "High" if prediction == 1 else "Low",
        "threshold": THRESHOLD
    }