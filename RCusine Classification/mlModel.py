import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from Prepocessor import preprocess_data

def train_models():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, "Dataset .csv"))

    X, y, scaler, mlb, df_clean = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))
    xgb = MultiOutputClassifier(XGBClassifier(n_estimators=150, eval_metric='logloss'))

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    rf_score = rf.score(X_test, y_test)
    xgb_score = xgb.score(X_test, y_test)

    os.makedirs("models", exist_ok=True)

    joblib.dump(rf, "models/rf_model.pkl")
    joblib.dump(xgb, "models/xgb_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(mlb, "models/mlb.pkl")

    best = "xgb" if xgb_score > rf_score else "rf"

    with open("models/best_model.txt", "w") as f:
        f.write(best)

    print(f"✅ Best Model: {best}")

if __name__ == "__main__":
    train_models()