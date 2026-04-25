import os
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

def train_tokenizer():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, "Dataset .csv"))

    texts = df["Cuisines"].astype(str)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)

    os.makedirs("models", exist_ok=True)

    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("✅ Tokenizer saved")

if __name__ == "__main__":
    train_tokenizer()