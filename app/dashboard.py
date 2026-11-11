from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Customer Churn Analysis Project", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

ACCENT = "#697CF7"
RED = "#F97373"
GREEN = "#2FBF9B"
NAVY = "#15223B"
THRESHOLD = 0.45
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CANDIDATES = [
    PROJECT_ROOT / "dataset" / "Telco_customer_churn.xlsx",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family: 'DM Sans', sans-serif;}
[data-testid="stAppViewContainer"] {background: #F6F8FC;}
[data-testid="stSidebar"] {background: #14213A;}
[data-testid="stSidebar"] * {color:#E8EEF9 !important;}
[data-testid="stSidebar"] div[role="radiogroup"] label {border-radius: 10px; padding: 7px 10px;}
.block-container {max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem;}
h1, h2, h3 {color: #17243C; letter-spacing: -.025em;}
div[data-testid="stMetric"] {background: white; border:1px solid #E8EDF5; padding:17px 19px; border-radius:14px; box-shadow:0 6px 24px rgba(24,43,80,.035);}
[data-testid="stMetricLabel"] {color: #68758A;}
[data-testid="stMetricValue"] {font-weight:700; color:#14223A;}
div[data-testid="stForm"] {background:white; border:1px solid #E8EDF5; border-radius:14px; padding:20px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#E8EDF5;}
.page-sub {color:#748199; margin-top:-8px; margin-bottom:16px;}
.mini-label {font-weight:700; font-size: .83rem; letter-spacing:.09em; color:#78869C; text-transform:uppercase;}
.insight {padding:14px 18px; border-left:4px solid #697CF7; border-radius:8px; background:#EAEEFF; color:#233352; margin: 15px 0;}
</style>
""", unsafe_allow_html=True)


def title(text: str, subtitle: str):
    st.title(text)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)

COLUMN_MAPPING = {
    "CustomerID": "customerID",
    "Senior Citizen": "SeniorCitizen",
    "Tenure Months": "tenure",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Churn Value": "Churn",
}


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMN_MAPPING)

    return df


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "TotalCharges" in df:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "MonthlyCharges" in df:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    if "tenure" in df:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    if "Churn" in df:
        df["Churn"] = df["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
    return df


def segment_chart(df: pd.DataFrame, segment: str, horizontal: bool = True):
    grouped = df.groupby(segment, dropna=False)["Churn"].agg(["mean", "size"]).reset_index()
    grouped = grouped.rename(columns={"mean": "churn_rate", "size": "customers"}).sort_values("churn_rate")
    grouped["churn_rate"] *= 100
    if horizontal:
        fig = px.bar(grouped, x="churn_rate", y=segment, orientation="h", text="churn_rate",
                     hover_data={"customers": True, "churn_rate": ":.1f"}, color_discrete_sequence=[ACCENT])
        fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="Churn rate (%)", yaxis_title="", xaxis_range=[0, max(55, grouped["churn_rate"].max() * 1.22)])
    else:
        fig = px.bar(grouped, x=segment, y="churn_rate", text="churn_rate",
                     hover_data={"customers": True}, color_discrete_sequence=[ACCENT])
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig.update_layout(yaxis_title="Churn rate (%)", xaxis_title="")
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=12, r=15, t=22, b=12),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      font=dict(family="DM Sans"))
    return fig


def churn_donut(df: pd.DataFrame):
    counts = df["Churn"].value_counts().reindex([0, 1], fill_value=0)
    fig = go.Figure(go.Pie(labels=["Retained", "Churned"], values=counts,
                           hole=.72, marker=dict(colors=[GREEN, RED]), textinfo="percent",
                           hovertemplate="%{label}: %{value:,} customers<extra></extra>"))
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=15, r=15, t=15, b=15),
                      showlegend=True, legend=dict(orientation="h", y=-.1, x=.23),
                      paper_bgcolor="rgba(0,0,0,0)", annotations=[dict(text=f"{counts.sum():,}<br><span style='font-size:12px'>Customers</span>",
                                                                      x=.5, y=.5, font_size=26, showarrow=False)])
    return fig


def filtered_data(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("---")
        st.caption("ANALYSIS FILTERS")
        contract_values = sorted(df["Contract"].dropna().unique().tolist()) if "Contract" in df else []
        internet_values = sorted(df["InternetService"].dropna().unique().tolist()) if "InternetService" in df else []
        contract_selected = st.multiselect("Contract", contract_values, default=contract_values)
        internet_selected = st.multiselect("Internet service", internet_values, default=internet_values)
    if contract_values:
        df = df[df["Contract"].isin(contract_selected)]
    if internet_values:
        df = df[df["InternetService"].isin(internet_selected)]
    return df


def overview(df: pd.DataFrame):
    title("Customer Overview", "Monitor retention performance and identify where customer churn is concentrated.")
    if df.empty:
        st.warning("Your filters returned zero customers. Select at least one category in each filter.")
        return
    valid = df.dropna(subset=["Churn"])
    if valid.empty:
        st.error("No valid Churn labels in selected data.")
        return
    total = len(valid)
    churned = int(valid["Churn"].sum())
    retained = total - churned
    a, b, c, d = st.columns(4)
    a.metric("Total Customers", f"{total:,}")
    b.metric("Churned Customers", f"{churned:,}")
    c.metric("Retained Customers", f"{retained:,}")
    d.metric("Churn Rate", f"{churned/total:.2%}")
    st.write("")
    col1, col2 = st.columns([1, 1.5], gap="large")
    with col1:
        st.subheader("Customer Retention Split")
        st.plotly_chart(churn_donut(valid), use_container_width=True)
    with col2:
        st.subheader("Churn by Contract")
        if "Contract" in valid:
            st.plotly_chart(segment_chart(valid, "Contract"), use_container_width=True)
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Churn by Internet Service")
        if "InternetService" in valid:
            st.plotly_chart(segment_chart(valid, "InternetService"), use_container_width=True)
    with right:
        st.subheader("Monthly Charges vs Churn")
        if "MonthlyCharges" in valid:
            fig = px.box(valid.assign(Status=valid["Churn"].map({0: "Retained", 1: "Churned"})),
                         x="Status", y="MonthlyCharges", color="Status",
                         color_discrete_map={"Retained": GREEN, "Churned": RED}, points=False)
            fig.update_layout(template="plotly_white", height=330, showlegend=False,
                              margin=dict(l=12, r=12, t=22, b=12), xaxis_title="", yaxis_title="Monthly charges ($)")
            st.plotly_chart(fig, use_container_width=True)
    if "MonthlyCharges" in valid:
        observed_churn_monthly = valid.loc[valid["Churn"] == 1, "MonthlyCharges"].sum()
        st.markdown(f'<div class="insight"><b>Historical churn exposure:</b> ${observed_churn_monthly:,.0f} in combined monthly charges from customers labeled churned. This is descriptive, not a forecast of future revenue loss.</div>', unsafe_allow_html=True)


def analysis(df: pd.DataFrame):
    title("Churn Analysis", "Explore churn patterns by customer demographics, services, and account history.")
    if df.empty:
        st.warning("No data matches the current filters.")
        return
    valid = df.dropna(subset=["Churn"])
    candidates = [x for x in ["Contract", "PaymentMethod", "InternetService", "PaperlessBilling",
                              "OnlineSecurity", "TechSupport", "SeniorCitizen", "Partner", "Dependents"] if x in valid]
    if candidates:
        field = st.selectbox("Compare customer segments", candidates)
        st.plotly_chart(segment_chart(valid, field), use_container_width=True)
    a, b = st.columns(2)
    if "tenure" in valid:
        with a:
            st.subheader("Tenure Distribution")
            fig = px.histogram(valid.assign(Status=valid["Churn"].map({0:"Retained",1:"Churned"})),
                               x="tenure", color="Status", barmode="overlay", nbins=24,
                               color_discrete_map={"Retained":GREEN,"Churned":RED}, opacity=.72)
            fig.update_layout(template="plotly_white", height=350, xaxis_title="Months as customer", yaxis_title="Customers")
            st.plotly_chart(fig, use_container_width=True)
    if "MonthlyCharges" in valid:
        with b:
            st.subheader("Monthly Charges Distribution")
            fig = px.histogram(valid.assign(Status=valid["Churn"].map({0:"Retained",1:"Churned"})),
                               x="MonthlyCharges", color="Status", barmode="overlay", nbins=25,
                               color_discrete_map={"Retained":GREEN,"Churned":RED}, opacity=.72)
            fig.update_layout(template="plotly_white", height=350, xaxis_title="Monthly charges ($)", yaxis_title="Customers")
            st.plotly_chart(fig, use_container_width=True)
    with st.expander("View filtered data"):
        st.dataframe(valid, use_container_width=True, hide_index=True)
        st.download_button("Download filtered DATASET", valid.to_csv(index=False), "filtered_customers.DATASET", "text/csv")


def prediction_inputs():
    """Names and categorical values follow the original IBM Telco DATASET."""
    with st.form("prediction_form"):
        st.subheader("Customer Profile")
        a, b, c = st.columns(3)
        with a:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with b:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        with c:
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=10000.0, value=70.0, step=1.0)
            total_charges = st.number_input("Total charges ($)", min_value=0.0, max_value=1000000.0, value=840.0, step=10.0,
                                            help="Enter the actual billing total. Do not approximate for a real prediction.")
        submitted = st.form_submit_button("Analyze Churn Risk", type="primary", use_container_width=True)
    payload = {"gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
               "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
               "MultipleLines": multiple, "InternetService": internet, "OnlineSecurity": online_security,
               "OnlineBackup": online_backup, "DeviceProtection": device_protection,
               "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
               "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
               "MonthlyCharges": monthly, "TotalCharges": total_charges}
    return submitted, payload


def extract_probability(response: dict):
    """Accept a few common response shapes; adjust to match your actual FastAPI schema."""
    for key in ["churn_probability", "probability", "churn_proba", "churnProbability"]:
        if key in response:
            value = response[key]
            if isinstance(value, dict):
                value = value.get("churn", value.get("Yes", value.get("1")))
            prob = float(value)
            if 0 <= prob <= 1:
                return prob
    raise ValueError("Response has no recognized churn probability (expected a numeric 0–1 value).")


def predict(api_url: str):
    title("Predict Customer Churn", "Use the saved machine-learning model to estimate churn risk for an individual customer.")
    st.info("This form sends the original Telco feature names to your FastAPI /predict endpoint. If your Pydantic schema uses other names or expects preprocessed fields, map the payload accordingly.")
    submitted, payload = prediction_inputs()
    if not submitted:
        return
    if payload["PhoneService"] == "No":
        payload["MultipleLines"] = "No phone service"
    elif payload["MultipleLines"] == "No phone service":
        st.error("Multiple lines cannot be 'No phone service' when phone service is enabled.")
        return
    internet_children = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    if payload["InternetService"] == "No":
        for key in internet_children:
            payload[key] = "No internet service"
    elif any(payload[key] == "No internet service" for key in internet_children):
        st.error("Internet add-ons cannot be 'No internet service' when internet service is enabled.")
        return
    try:
        result = requests.post(api_url, json=payload, timeout=15)
        result.raise_for_status()
        data = result.json()
        prob = extract_probability(data)
    except requests.exceptions.RequestException as exc:
        st.error(f"FastAPI request failed: {exc}")
        st.caption("Check that FastAPI is running and that the API URL and request schema match.")
        return
    except (ValueError, TypeError) as exc:
        st.error(f"Could not interpret FastAPI response: {exc}")
        st.json(data if 'data' in locals() else {})
        return
    high_risk = prob >= THRESHOLD
    st.divider()
    a, b, c = st.columns(3)
    a.metric("Predicted Churn Probability", f"{prob:.1%}")
    b.metric("Decision Threshold", f"{THRESHOLD:.0%}")
    c.metric("Classification", "At risk" if high_risk else "Lower risk")
    fig = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100,
        number={"suffix":"%", "font":{"size":44}},
        gauge={"axis":{"range":[0,100]}, "bar":{"color":RED if high_risk else GREEN},
               "steps":[{"range":[0,45],"color":"#E6F6F0"}, {"range":[45,100],"color":"#FDEBEC"}],
               "threshold":{"line":{"color":"#17243C","width":3},"value":45}}))
    fig.update_layout(height=320, margin=dict(l=30,r=30,t=25,b=15), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.warning("Higher-risk customer: consider an appropriate retention outreach or service review." if high_risk else
               "Lower-risk customer: maintain service quality and monitor for changes.")
    st.caption("This is a statistical prediction, not a certainty. The 45% threshold is a decision rule; probability calibration has not been established here.")


def model_info():
    title("Model Performance", "Transparent documentation of the deployed churn classification model.")
    a, b, c = st.columns(3)
    a.metric("Algorithm", "Logistic Regression")
    b.metric("ROC-AUC", "0.8491")
    c.metric("Decision Threshold", "0.45")
    st.markdown("### How predictions work")
    st.markdown("**Customer form → FastAPI `/predict` → saved preprocessing/model pipeline → churn probability → threshold-based classification.**")
    st.markdown("### What the score means")
    st.write("ROC-AUC measures ranking performance across classification thresholds. It does not establish precision, recall, class-specific costs, or probability calibration. A 0.45 threshold means predicted probability ≥ 0.45 is classified as 'at risk'.")
    st.markdown("### Evaluation details to include before publishing")
    st.write("Add the confusion matrix, precision, recall, F1, precision-recall curve, calibration plot, validation method, test set size, and feature coefficients from your actual evaluation outputs. Avoid manufacturing these values.")
    st.markdown("### Data caveat")
    st.write("This demo is trained on historical IBM Telco sample data. Predictions should not be represented as production-validated forecasts for a live telecom operator.")


with st.sidebar:
    st.markdown("## 📈 Customer Churn Analysis Project")
    st.caption("TELCO INTELLIGENCE PLATFORM")
    st.caption("BY SAMA NEMKUL SHRESHTA")
    st.divider()
    page = st.radio("Workspace", ["Overview", "Churn Analysis", "Predict Churn", "Model Performance"], label_visibility="collapsed")
    st.divider()
    st.caption("MODEL CONFIGURATION")
    st.write("**Logistic Regression**")
    st.caption("ROC-AUC · 0.8491  |  Threshold · 0.45")
    st.divider()
    api_url = st.text_input("FastAPI prediction endpoint", value=os.getenv("CHURN_API_URL", "http://localhost:8501/predict"))

if page == "Predict Churn":
    predict(api_url)
elif page == "Model Performance":
    model_info()
else:
    data_file = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_file is not None:
        raw = load_dataset(str(data_file))
    else:
        with st.sidebar:
            uploaded = st.file_uploader("Upload IBM Telco DATASET", type=["DATASET"])
        raw = pd.read_dataset(uploaded) if uploaded is not None else None
    if raw is None:
        title("Customer Intelligence", "Add your IBM Telco DATASET to unlock live analytics.")
        st.info("Place the DATASET in `data/WA_Fn-UseC_-Telco-Customer-Churn.DATASET`, or upload it using the sidebar.")
        st.stop()
    required = ["Churn", "Contract", "InternetService"]
    missing = [key for key in required if key not in raw.columns]
    if missing:
        st.error(f"Missing columns: {missing}. Upload the original IBM Telco churn DATASET.")
        st.stop()
    filtered = filtered_data(prepare_data(raw))
    overview(filtered) if page == "Overview" else analysis(filtered)