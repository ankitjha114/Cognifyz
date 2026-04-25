import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

def preprocess_data(df):

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

    df["Cuisines"] = df["Cuisines"].apply(lambda x: [i.strip() for i in x.split(",")])

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["Cuisines"])

    X = df.drop("Cuisines", axis=1)

    X["Has Table booking"] = X["Has Table booking"].map({"No":0, "Yes":1})
    X["Has Online delivery"] = X["Has Online delivery"].map({"No":0, "Yes":1})

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, mlb, df