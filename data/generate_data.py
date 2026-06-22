import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

gender         = np.random.choice(["Male", "Female"], N, p=[0.65, 0.35])
married        = np.random.choice(["Yes", "No"],      N, p=[0.65, 0.35])
dependents     = np.random.choice(["0", "1", "2", "3+"], N, p=[0.57, 0.17, 0.16, 0.10])
education      = np.random.choice(["Graduate", "Not Graduate"], N, p=[0.78, 0.22])
self_employed  = np.random.choice(["Yes", "No"], N, p=[0.14, 0.86])

applicant_income   = np.random.lognormal(mean=8.5, sigma=0.6, size=N).astype(int)
coapplicant_income = np.where(
    married == "Yes",
    np.random.lognormal(mean=7.5, sigma=0.8, size=N).astype(int),
    0,
)
loan_amount       = (applicant_income * np.random.uniform(0.8, 4.0, N)).astype(int)
loan_term         = np.random.choice([120, 180, 240, 300, 360, 480], N, p=[0.03, 0.04, 0.05, 0.08, 0.70, 0.10])
credit_history    = np.random.choice([0, 1], N, p=[0.18, 0.82])
property_area     = np.random.choice(["Urban", "Semiurban", "Rural"], N, p=[0.33, 0.38, 0.29])

# Approval logic (realistic, not purely random)
score = (
    (credit_history == 1) * 0.40
    + (education == "Graduate") * 0.10
    + (applicant_income > 5000) * 0.15
    + (coapplicant_income > 1000) * 0.10
    + (loan_amount < applicant_income * 2) * 0.10
    + (property_area == "Semiurban") * 0.05
    + (self_employed == "No") * 0.05
    + np.random.uniform(0, 0.15, N)  # noise
)
loan_status = np.where(score >= 0.55, "Y", "N")

df = pd.DataFrame({
    "Gender":            gender,
    "Married":           married,
    "Dependents":        dependents,
    "Education":         education,
    "Self_Employed":     self_employed,
    "ApplicantIncome":   applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount":        loan_amount,
    "Loan_Amount_Term":  loan_term,
    "Credit_History":    credit_history,
    "Property_Area":     property_area,
    "Loan_Status":       loan_status,
})

out = "data/loan_data.csv"
df.to_csv(out, index=False)
print(f"Dataset saved → {out}  ({N} rows, approval rate: {(loan_status=='Y').mean():.1%})")