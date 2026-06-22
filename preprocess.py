import pandas as pd
import numpy as np


# ── Categorical columns and their fill strategy 
CAT_COLS   = ["Gender", "Married", "Dependents", "Self_Employed", "Property_Area"]
NUM_COLS   = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
CREDIT_COL = "Credit_History"

# Ordinal encoding maps
GENDER_MAP      = {"Male": 0, "Female": 1}
MARRIED_MAP     = {"No": 0, "Yes": 1}
DEPENDENTS_MAP  = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION_MAP   = {"Graduate": 0, "Not Graduate": 1}
EMPLOYED_MAP    = {"No": 0, "Yes": 1}
PROPERTY_MAP    = {"Rural": 0, "Semiurban": 1, "Urban": 2}


def clean_and_encode(df: pd.DataFrame, is_train: bool = True):
    df = df.copy()

    # ── Fill missing values 
    df["Gender"]        = df["Gender"].fillna("Male")
    df["Married"]       = df["Married"].fillna("Yes")
    df["Dependents"]    = df["Dependents"].fillna("0")
    df["Self_Employed"] = df["Self_Employed"].fillna("No")
    df["Credit_History"]= df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    medians = {}
    for col in NUM_COLS:
        medians[col] = df[col].median()
        df[col]      = df[col].fillna(medians[col])

    # ── Engineer features 
    df["TotalIncome"]        = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Log_TotalIncome"]    = np.log1p(df["TotalIncome"])
    df["Log_LoanAmount"]     = np.log1p(df["LoanAmount"])
    df["DebtIncomeRatio"]    = df["LoanAmount"] / (df["TotalIncome"] + 1)

    # ── Encode categoricals 
    df["Gender"]        = df["Gender"].map(GENDER_MAP)
    df["Married"]       = df["Married"].map(MARRIED_MAP)
    df["Dependents"]    = df["Dependents"].map(DEPENDENTS_MAP)
    df["Education"]     = df["Education"].map(EDUCATION_MAP)
    df["Self_Employed"] = df["Self_Employed"].map(EMPLOYED_MAP)
    df["Property_Area"] = df["Property_Area"].map(PROPERTY_MAP)

    FEATURE_COLS = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "Log_TotalIncome", "Log_LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Property_Area", "DebtIncomeRatio",
    ]

    X = df[FEATURE_COLS]

    if is_train:
        y = (df["Loan_Status"] == "Y").astype(int)
        return X, y, medians

    return X, None, medians