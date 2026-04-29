import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Dataset .csv")

df = pd.read_csv(file_path)

print(df)
print(df.info())
print(df.describe())
print(df.duplicated().sum())
print(df.isna().sum())
print(df.columns)

df.groupby("City")["Average Cost for two"].mean()\
    .sort_values(ascending=False).head()\
    .plot(kind="bar")
plt.title("Average cost for two per city, top 5")
plt.xlabel("City")
plt.ylabel("Amount")
plt.show()

data = df.groupby("Cuisines")["Votes"].sum().reset_index()
print(data)

sns.pairplot(df[["Average Cost for two", "Aggregate rating", "Votes"]])
plt.show()

df["Has Online delivery"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Has Online Delivery")
plt.ylabel("")
plt.show()

df = df[df["Rating text"] != "Not rated"]

df["Rating text"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Rating Distribution")
plt.ylabel("")
plt.show()

X = df[["Average Cost for two", "Has Table booking", "Has Online delivery", "Price range"]].copy()
y = df["Aggregate rating"]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X["Has Table booking"] = le.fit_transform(X["Has Table booking"])
X["Has Online delivery"] = le.fit_transform(X["Has Online delivery"])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2:", r2_score(y_true, y_pred))

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

models = {}

lr = LinearRegression()
lr.fit(X_train, y_train)
models["Linear Regression"] = lr.predict(X_test)

svr = SVR()
svr.fit(X_train, y_train)
models["SVR"] = svr.predict(X_test)

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
models["Decision Tree"] = dt.predict(X_test)

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
models["Random Forest"] = rf.predict(X_test)

knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
models["KNN"] = knn.predict(X_test)

ada = AdaBoostRegressor()
ada.fit(X_train, y_train)
models["AdaBoost"] = ada.predict(X_test)

xgb = XGBRegressor(n_estimators=200, learning_rate=0.1)
xgb.fit(X_train, y_train)
models["XGBoost"] = xgb.predict(X_test)

results = {}

for name, preds in models.items():
    r2 = r2_score(y_test, preds)
    results[name] = r2
    evaluate_model(name, y_test, preds)

best_model_name = max(results, key=results.get)
print("\nBest Model:", best_model_name)

best_model = None

if best_model_name == "Linear Regression":
    best_model = lr
elif best_model_name == "SVR":
    best_model = svr
elif best_model_name == "Decision Tree":
    best_model = dt
elif best_model_name == "Random Forest":
    best_model = rf
elif best_model_name == "KNN":
    best_model = knn
elif best_model_name == "AdaBoost":
    best_model = ada
else:
    best_model = xgb

joblib.dump(best_model, os.path.join(BASE_DIR, "best_model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

print("Best model saved successfully!")

result_df = pd.DataFrame(list(results.items()), columns=["Model", "R2 Score"])

plt.figure(figsize=(10,5))
plt.bar(result_df["Model"], result_df["R2 Score"])
plt.xticks(rotation=30)
plt.title("Model Comparison (R2 Score)")
plt.show()
