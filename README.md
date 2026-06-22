# 🏦 Loan Approval Predictor — ML Project

A complete end-to-end machine learning project that predicts whether a loan
application will be approved, with a Flask web interface.

---

## 📁 Project Structure

```
loan_approval/
├── data/
│   ├── generate_data.py      # Synthetic dataset generator
│   └── loan_data.csv         # Generated after running generate_data.py
├── models/
│   ├── loan_model.pkl        # Saved model (after training)
│   ├── model_meta.json       # Accuracy / AUC / feature importances
│   └── roc_curve.png         # ROC curve plot
├── static/
│   ├── css/style.css         # Stylesheet
│   └── js/main.js            # Frontend JavaScript
├── templates/
│   └── index.html            # Jinja2 HTML template
├── preprocess.py             # Data cleaning & feature engineering
├── train.py                  # Model training & evaluation
├── app.py                    # Flask web application
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Generate dataset
```bash
python data/generate_data.py
```
Creates `data/loan_data.csv` with 1,000 synthetic loan records.

### 3 — Train the model
```bash
python train.py
```
Trains a Random Forest, prints evaluation metrics, saves the model to `models/`.

### 4 — Start the web app
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

---

## 🤖 ML Details

| Component | Choice |
|-----------|--------|
| Algorithm | Random Forest Classifier |
| Features | 11 engineered features |
| Train/Test split | 80/20, stratified |
| Cross-validation | 5-fold |
| Evaluation | Accuracy, ROC-AUC, Classification Report |

### Features Used
| Feature | Description |
|---------|-------------|
| Log_TotalIncome | log(applicant + co-applicant income) |
| Log_LoanAmount | log(loan amount in thousands) |
| DebtIncomeRatio | loan_amount / total_income |
| Credit_History | 1 = good, 0 = poor |
| Loan_Amount_Term | Loan tenure in months |
| Property_Area | Rural=0, Semiurban=1, Urban=2 |
| Education | Graduate=0, Not Graduate=1 |
| Married | No=0, Yes=1 |
| Self_Employed | No=0, Yes=1 |
| Dependents | 0, 1, 2, 3 |
| Gender | Male=0, Female=1 |

---

## 🌐 API Endpoint

### POST `/predict`
```json
{
  "gender": "Male",
  "married": "Yes",
  "dependents": "0",
  "education": "Graduate",
  "self_employed": "No",
  "applicant_income": 5000,
  "coapplicant_income": 2000,
  "loan_amount": 150,
  "loan_term": 360,
  "credit_history": 1,
  "property_area": "Urban"
}
```

**Response:**
```json
{
  "approved": true,
  "probability": 87.3,
  "confidence": 87.3,
  "verdict": "✅ Loan Approved",
  "message": "Strong application — very likely to be approved."
}
```

### GET `/model-info`
Returns model accuracy, AUC, feature importances.

---

## 📊 Expected Results
- Test Accuracy: ~82–87%
- ROC-AUC Score: ~0.88–0.92
- 5-Fold CV Accuracy: ~83–86%

---

*Built for educational purposes. Not for real financial decisions.*
