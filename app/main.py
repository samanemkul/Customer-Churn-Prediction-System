
import pandas as pd

from fastapi import FastAPI
from app.predictor import predict_churn

app = FastAPI()

from pydantic import BaseModel


class CustomerSchema(BaseModel):
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    
@app.post("/predict")
def predict(customer_data: CustomerSchema):
    return predict_churn(
        pd.DataFrame([customer_data.model_dump()])
        
    )