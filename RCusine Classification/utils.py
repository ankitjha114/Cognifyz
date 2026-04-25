import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_all():

    with open(os.path.join(BASE_DIR, "models/best_model.txt")) as f:
        best = f.read().strip()

    model_path = "xgb_model.pkl" if best == "xgb" else "rf_model.pkl"

    model = joblib.load(os.path.join(BASE_DIR, "models", model_path))
    scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
    mlb = joblib.load(os.path.join(BASE_DIR, "models/mlb.pkl"))

    return model, scaler, mlb, best


def get_probabilities(model, X):

    probs = model.predict_proba(X)

    final = []
    for p in probs:
        if p.shape[1] == 2:
            final.append(p[0][1])
        else:
            final.append(p[0][0])

    final = np.array(final)
    return final / final.sum()