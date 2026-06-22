import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from preprocess import clean_and_encode

app = Flask(__name__)

MODEL_PATH = "models/loan_model.pkl"
META_PATH  = "models/model_meta.json"

# ── Load model once at startup 
_artifact = None
_meta     = {}

def load_model():
    global _artifact, _meta
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please run: python data/generate_data.py && python train.py"
        )
    with open(MODEL_PATH, "rb") as f:
        _artifact = pickle.load(f)
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _meta = json.load(f)


@app.route("/")
def index():
    return render_template("index.html", meta=_meta)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Build a one-row DataFrame from the incoming JSON
    row = {
        "Gender":            data.get("gender",           "Male"),
        "Married":           data.get("married",          "Yes"),
        "Dependents":        data.get("dependents",       "0"),
        "Education":         data.get("education",        "Graduate"),
        "Self_Employed":     data.get("self_employed",    "No"),
        "ApplicantIncome":   float(data.get("applicant_income",   5000)),
        "CoapplicantIncome": float(data.get("coapplicant_income", 0)),
        "LoanAmount":        float(data.get("loan_amount",        150)),
        "Loan_Amount_Term":  float(data.get("loan_term",          360)),
        "Credit_History":    float(data.get("credit_history",     1)),
        "Property_Area":     data.get("property_area",    "Urban"),
    }

    df = pd.DataFrame([row])
    X, _, _ = clean_and_encode(df, is_train=False)

    model  = _artifact["model"]
    prob   = model.predict_proba(X)[0]
    label  = model.predict(X)[0]

    result = {
        "approved":    bool(label == 1),
        "probability": round(float(prob[1]) * 100, 1),
        "confidence":  round(float(max(prob)) * 100, 1),
        "verdict":     "✅ Loan Approved" if label == 1 else "❌ Loan Rejected",
        "message":     _get_message(label, prob[1]),
    }
    return jsonify(result)


@app.route("/model-info")
def model_info():
    return jsonify(_meta)


def _get_message(label, prob):
    if label == 1:
        if prob >= 0.85:
            return "Strong application — very likely to be approved."
        return "Good application — approval is probable."
    else:
        if prob <= 0.25:
            return "Weak application — low approval probability."
        return "Borderline application — improve credit history or income."


if __name__ == "__main__":
    load_model()
    print("  Loan Approval App running → http://127.0.0.1:5000")
    app.run(debug=True)