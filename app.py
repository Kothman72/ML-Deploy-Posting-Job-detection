from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "model" / "fake_job_posting_pipeline.joblib"


st.set_page_config(page_title="Fake Job Posting Detector")
st.title("Fake Job Posting Detector")
st.write(
    "Enter the job-posting details to estimate whether the posting may be fraudulent. "
    "Treat the result as a screening aid, not a definitive judgment."
)


@st.cache_resource
def load_model_bundle(model_path: str, modified_time: float) -> dict:
    return joblib.load(model_path)


if not MODEL_PATH.is_file():
    st.error(f"Model file not found: `{MODEL_PATH}`")
    st.info(
        "Train the model by running the notebook's "
        "'Leak-safe word + character TF-IDF model' cell, then place the generated "
        "`model` folder beside `app.py`."
    )
    st.stop()

bundle = load_model_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
required_bundle_keys = {
    "pipeline",
    "threshold",
    "feature_columns",
    "text_columns",
    "category_columns",
    "numeric_columns",
}
missing_keys = required_bundle_keys.difference(bundle)
if missing_keys:
    raise ValueError(f"Saved model bundle is missing keys: {sorted(missing_keys)}")


def prepare_features(raw_values: dict, model_bundle: dict) -> pd.DataFrame:
    raw = pd.DataFrame([raw_values])
    text_columns = model_bundle["text_columns"]
    text_data = {}

    for column in text_columns:
        if column not in raw:
            raw[column] = ""
        text = raw[column].fillna("").astype(str)
        text_data[column] = text
        raw[f"{column}_log_chars"] = np.log1p(text.str.len())
        raw[f"{column}_log_words"] = np.log1p(text.str.split().str.len())

    if text_data:
        raw["combined_text"] = pd.DataFrame(text_data, index=raw.index).agg(
            " ".join, axis=1
        )
    else:
        raw["combined_text"] = ""

    for column in model_bundle["category_columns"]:
        if column in raw:
            blank_categories = raw[column].astype(str).str.strip().eq("")
            raw.loc[blank_categories, column] = np.nan

    return raw.reindex(columns=model_bundle["feature_columns"])


with st.form("job_posting_form"):
    st.subheader("Job details")
    title = st.text_input("Job title")
    description = st.text_area("Description")
    company_profile = st.text_area("Company profile")
    requirements = st.text_area("Requirements")
    benefits = st.text_area("Benefits")

    st.subheader("Additional information")
    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("Location")
        department = st.text_input("Department")
        salary_range = st.text_input("Salary range", placeholder="e.g. 40000-60000")
        employment_type = st.text_input("Employment type")
        required_experience = st.text_input("Required experience")
    with col2:
        required_education = st.text_input("Required education")
        industry = st.text_input("Industry")
        function = st.text_input("Job function")
        telecommuting = st.selectbox("Telecommuting", ["No", "Yes"])
        has_company_logo = st.selectbox("Company logo", ["No", "Yes"])
        has_questions = st.selectbox("Application questions", ["No", "Yes"])

    submitted = st.form_submit_button("Check posting")

if submitted:
    raw_values = {
        "title": title,
        "description": description,
        "company_profile": company_profile,
        "requirements": requirements,
        "benefits": benefits,
        "location": location,
        "department": department,
        "salary_range": salary_range,
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function,
        "telecommuting": int(telecommuting == "Yes"),
        "has_company_logo": int(has_company_logo == "Yes"),
        "has_questions": int(has_questions == "Yes"),
    }
    features = prepare_features(raw_values, bundle)
    pipeline = bundle["pipeline"]
    classifier = pipeline.named_steps["classifier"]
    fraud_class_index = list(classifier.classes_).index(1)
    fraud_probability = float(pipeline.predict_proba(features)[0, fraud_class_index])
    threshold = float(bundle["threshold"])

    st.metric("Estimated fraud probability", f"{fraud_probability:.1%}")
    st.caption(f"Decision threshold from validation: {threshold:.3f}")
    if fraud_probability >= threshold:
        st.error("Prediction: potentially fraudulent")
    else:
        st.success("Prediction: likely legitimate")
