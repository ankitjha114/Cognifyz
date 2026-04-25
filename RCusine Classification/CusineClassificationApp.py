# import streamlit as st
# import pandas as pd
# import numpy as np
# import shap

# from utils import load_all, get_probabilities

# st.set_page_config(layout="wide")
# st.title("🍽️ Cuisine AI (XGBoost + Explainable AI)")

# model, scaler, mlb = load_all()

# # ---------------- INPUT ---------------- #
# col1, col2 = st.columns(2)

# with col1:
#     cost = st.slider("Average Cost", 0, 10000, 500)

# with col2:
#     votes = st.slider("Votes", 0, 5000, 100)

# price = st.slider("Price Range", 1, 4, 2)

# table = st.selectbox("Table Booking", ["No", "Yes"])
# delivery = st.selectbox("Online Delivery", ["No", "Yes"])

# if st.button("Predict"):

#     table = 1 if table == "Yes" else 0
#     delivery = 1 if delivery == "Yes" else 0

#     input_df = pd.DataFrame([[cost, votes, price, table, delivery]],
#         columns=[
#             "Average Cost for two",
#             "Votes",
#             "Price range",
#             "Has Table booking",
#             "Has Online delivery"
#         ])

#     input_scaled = scaler.transform(input_df)

#     # -------- HYBRID (ML core) -------- #
#     probs = get_probabilities(model, input_scaled)

#     top_idx = np.argsort(probs)[::-1][:5]

#     st.subheader("🏆 Top 5 Cuisines")

#     for i in top_idx:
#         st.write(f"{mlb.classes_[i]}")

#     # -------- CONFIDENCE -------- #
#     st.subheader("📊 Confidence")

#     for i in top_idx:
#         st.progress(float(probs[i]))
#         st.write(f"{mlb.classes_[i]} → {probs[i]:.2f}")

#     # ---------------- SHAP EXPLAINABILITY ---------------- #
#     import shap

#     st.subheader("🧠 Why this prediction?")

#     try:
#         # ✅ Extract raw XGBoost model
#         base_model = model.estimators_[0]

#         # ✅ Use safe explainer
#         explainer = shap.Explainer(base_model)

#         shap_values = explainer(input_scaled)

#         feature_names = [
#             "Average Cost",
#             "Votes",
#             "Price Range",
#             "Table Booking",
#             "Online Delivery"
#         ]

#         st.write("Feature Impact:")

#         for i, val in enumerate(shap_values.values[0]):
#             st.write(f"{feature_names[i]} → {val:.4f}")

#     except Exception as e:
#         st.warning("SHAP explanation failed. Showing fallback importance.")

#         # 🔥 Fallback (feature importance)
#         importances = base_model.feature_importances_

#         feature_names = [
#             "Average Cost",
#             "Votes",
#             "Price Range",
#             "Table Booking",
#             "Online Delivery"
#         ]

#         for i in range(len(importances)):
#             st.write(f"{feature_names[i]} → {importances[i]:.4f}")







import streamlit as st
import pandas as pd
import numpy as np
import shap

from utils import load_all, get_probabilities
from recommender import recommend

st.set_page_config(layout="wide")
st.title("🍽️ Cuisine AI System")

model, scaler, mlb, model_name = load_all()

st.write(f"Using Model: **{model_name.upper()}**")

# INPUT
cost = st.slider("Average Cost", 0, 10000, 500)
votes = st.slider("Votes", 0, 5000, 100)
price = st.slider("Price Range", 1, 4, 2)

table = st.selectbox("Table Booking", ["No","Yes"])
delivery = st.selectbox("Online Delivery", ["No","Yes"])

if st.button("Predict"):

    table = 1 if table=="Yes" else 0
    delivery = 1 if delivery=="Yes" else 0

    X = pd.DataFrame([[cost, votes, price, table, delivery]],
        columns=[
            "Average Cost for two",
            "Votes",
            "Price range",
            "Has Table booking",
            "Has Online delivery"
        ])

    X_scaled = scaler.transform(X)

    probs = get_probabilities(model, X_scaled)

    top_idx = np.argsort(probs)[::-1][:5]

    st.subheader("🏆 Top Cuisines")

    selected = []
    for i in top_idx:
        cuisine = mlb.classes_[i]
        selected.append(cuisine)
        st.write(f"{cuisine} ({probs[i]:.2f})")
        st.progress(float(probs[i]))

    # ---------------- SHAP ---------------- #
    st.subheader("📊 SHAP Explanation")

    try:
        base_model = model.estimators_[0]
        explainer = shap.Explainer(base_model)
        shap_values = explainer(X_scaled)

        st.write("Feature Impact:")
        st.bar_chart(shap_values.values[0])

    except:
        st.warning("SHAP failed, showing fallback importance")
        st.bar_chart(base_model.feature_importances_)

    # ---------------- RECOMMENDER ---------------- #
    st.subheader("🍽️ Recommended Restaurants")

    df = pd.read_csv("Dataset .csv")

    recs = recommend(df, selected)

    for r in recs:
        st.write(f"{r['Restaurant Name']} - {r['Cuisines']}")