import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import os

def train_models():
    # Get correct path of Dataset.csv
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "Dataset .csv")

    df = pd.read_csv(file_path)

    # ---------------- CLEANING ---------------- #
    df = df[df["Rating text"] != "Not rated"]

    # Remove useless columns (VERY IMPORTANT)
    drop_cols = ["Restaurant ID", "Restaurant Name", "Address", "Locality", "Locality Verbose"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Handle missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Reduce extreme values (important for stability)
    if "Average Cost for two" in df.columns:
        df["Average Cost for two"] = np.log1p(df["Average Cost for two"])

    # ---------------- FEATURES ---------------- #
    X = df.drop(["Aggregate rating", "Rating text"], axis=1)
    y = df["Aggregate rating"]

    # Encode categorical safely
    X = pd.get_dummies(X, drop_first=True)

    # Remove constant columns (important fix)
    X = X.loc[:, (X != X.iloc[0]).any()]

    # ---------------- SPLIT ---------------- #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- SCALING ---------------- #
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- MODELS ---------------- #
    scores = {}

    # 🔹 Linear Regression (SAFE)
    try:
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)
        scores["Linear Regression"] = r2_score(y_test, y_pred_lr)
    except Exception as e:
        print("Linear Regression failed:", e)
        lr = None
        scores["Linear Regression"] = 0

    # 🔹 SVR
    svr = SVR(kernel="rbf", C=1)
    svr.fit(X_train_scaled, y_train)
    y_pred_svr = svr.predict(X_test_scaled)
    scores["SVR"] = r2_score(y_test, y_pred_svr)

    # 🔹 Random Forest (BEST)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)   # NOTE: no scaling needed
    y_pred_rf = rf.predict(X_test)
    scores["Random Forest"] = r2_score(y_test, y_pred_rf)

    return lr, svr, rf, scaler, X.columns, scores, df