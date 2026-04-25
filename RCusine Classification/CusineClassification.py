import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def train_cuisine_model():

    # ---------------- FILE PATH ---------------- #
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "Dataset .csv")   # ✅ FIXED

    df = pd.read_csv(file_path)

    # ---------------- BASIC EDA ---------------- #
    print(df.head())
    print(df.info())
    print(df.isna().sum())

    # ---------------- CLEANING ---------------- #
    df = df[df["Cuisines"].notna()]
    df = df[df["Rating text"] != "Not rated"]

    df = df[[
        "Cuisines",
        "Average Cost for two",
        "Votes",
        "Price range",
        "Has Table booking",
        "Has Online delivery"
    ]]

    df.dropna(inplace=True)

    # ---------------- TARGET ---------------- #
    df["Cuisines"] = df["Cuisines"].apply(lambda x: x.split(",")[0].strip())

    # ---------------- ENCODING ---------------- #
    le_target = LabelEncoder()
    df["Cuisines"] = le_target.fit_transform(df["Cuisines"])

    le = LabelEncoder()
    df["Has Table booking"] = le.fit_transform(df["Has Table booking"])
    df["Has Online delivery"] = le.fit_transform(df["Has Online delivery"])

    # ---------------- FEATURES ---------------- #
    X = df.drop("Cuisines", axis=1)
    y = df["Cuisines"]

    # ---------------- SCALING ---------------- #
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------- SPLIT ---------------- #
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ---------------- MODELS ---------------- #
    models = {}

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    svm = SVC()
    svm.fit(X_train, y_train)
    models["SVM"] = svm

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb

    # ---------------- EVALUATION ---------------- #
    results = {}

    for name, model in models.items():
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        results[name] = acc

        print(f"\n{name}")
        print("Accuracy:", acc)
        print(classification_report(y_test, preds))

    # ---------------- BEST MODEL ---------------- #
    best_model_name = max(results, key=results.get)
    print("\n🏆 Best Model:", best_model_name)

    best_model = models[best_model_name]

    # ---------------- SAVE ---------------- #
    joblib.dump(best_model, os.path.join(BASE_DIR, "best_cuisine_model.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler_cuisine.pkl"))
    joblib.dump(le_target, os.path.join(BASE_DIR, "label_encoder.pkl"))

    print("✅ Model saved successfully!")

    # ================= VISUALIZATIONS ================= #

    # 🎯 1. Model Comparison Bar Chart
    result_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])

    plt.figure()
    plt.bar(result_df["Model"], result_df["Accuracy"])
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=30)
    plt.show()

    # 🎯 2. Pie Chart (Cuisine Distribution)
    cuisine_counts = df["Cuisines"].value_counts().head(5)

    plt.figure()
    plt.pie(cuisine_counts, labels=cuisine_counts.index, autopct="%1.1f%%")
    plt.title("Top 5 Cuisine Distribution")
    plt.show()

    # 🎯 3. Confusion Matrix Heatmap
    preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    plt.figure()
    sns.heatmap(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # 🎯 4. Correlation Heatmap
    plt.figure()
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.show()

    return best_model, scaler, le_target


# ================= RUN ================= #
if __name__ == "__main__":
    print("🚀 Training started...")
    train_cuisine_model()
    print("✅ Done!")