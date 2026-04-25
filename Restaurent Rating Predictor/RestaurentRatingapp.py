import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import register_user, login_user, save_prediction
from model import train_models

st.set_page_config(page_title="Restaurant Rating Predictor", layout="wide")

# Load models
lr, svr, rf, scaler, columns, scores, df = train_models()

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# ---------------- AUTH ---------------- #
st.sidebar.title("🔐 Authentication")
choice = st.sidebar.selectbox("Choose", ["Login", "Register"])

if choice == "Register":
    st.subheader("Create Account")
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        if register_user(user, password):
            st.success("Account created!")
        else:
            st.error("User already exists")

elif choice == "Login":
    st.subheader("Login")
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(user, password):
            st.session_state.logged_in = True
            st.session_state.username = user
            st.success("Logged in!")
        else:
            st.error("Invalid credentials")

# ---------------- MAIN APP ---------------- #
if st.session_state.logged_in:

    st.title("🍽️ Restaurant Rating Predictor Dashboard")

    menu = st.sidebar.selectbox(
        "Navigation",
        ["Prediction", "Dashboard", "Model Comparison"]
    )

    # ---------------- 🔮 PREDICTION ---------------- #
    if menu == "Prediction":
        st.header("Predict Rating")

        cost = st.number_input("Average Cost for Two", min_value=0)
        votes = st.number_input("Votes", min_value=0)

        model_choice = st.selectbox(
            "Choose Model",
            ["Linear Regression", "SVR", "Random Forest"]
        )

        if st.button("Predict"):
            input_data = pd.DataFrame(
                [[cost, votes]],
                columns=["Average Cost for two", "Votes"]
            )

            for col in columns:
                if col not in input_data.columns:
                    input_data[col] = 0

            input_data = input_data[columns]
            input_scaled = scaler.transform(input_data)

            if model_choice == "Linear Regression" and lr is not None:
                prediction = lr.predict(input_scaled)[0]
            elif model_choice == "SVR":
                prediction = svr.predict(input_scaled)[0]
            else:
                prediction = rf.predict(input_scaled)[0]

            st.success(f"Predicted Rating: {round(prediction, 2)} ⭐")

            save_prediction(
                st.session_state.username,
                {"cost": cost, "votes": votes},
                prediction
            )

    # ---------------- 📊 DASHBOARD ---------------- #
    elif menu == "Dashboard":
        st.header("Data Insights")

        col1, col2 = st.columns(2)

        # Pie chart
        with col1:
            st.subheader("Rating Distribution")
            fig1, ax1 = plt.subplots()
            df["Rating text"].value_counts().plot(
                kind="pie", autopct="%1.1f%%", ax=ax1
            )
            st.pyplot(fig1)

        # Bar chart
        with col2:
            st.subheader("Top Cities by Cost")
            fig2, ax2 = plt.subplots()
            df.groupby("City")["Average Cost for two"]\
                .mean().sort_values(ascending=False).head()\
                .plot(kind="bar", ax=ax2)
            st.pyplot(fig2)

    # ---------------- 🤖 MODEL COMPARISON ---------------- #
    elif menu == "Model Comparison":
        st.header("Model Performance Comparison")

        st.write("R2 Scores of Models:")

        score_df = pd.DataFrame(
            list(scores.items()),
            columns=["Model", "R2 Score"]
        )

        st.dataframe(score_df)

        fig, ax = plt.subplots()
        ax.bar(score_df["Model"], score_df["R2 Score"])
        st.pyplot(fig)

    # Logout
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""