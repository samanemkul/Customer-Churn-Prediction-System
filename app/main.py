from fastapi import FastAPI
from app.predictor import predict_churn

app = FastAPI()

@app.post("/predict")
def predict(customer_data: CustomerSchema):
    return predict_churn(
        pd.DataFrame([customer_data.model_dump()])
    )