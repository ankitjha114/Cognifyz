import json
import hashlib

DB_FILE = "database.json"

def load_db():
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    db = load_db()

    for user in db["users"]:
        if user["username"] == username:
            return False

    db["users"].append({
        "username": username,
        "password": hash_password(password)
    })

    save_db(db)
    return True

def login_user(username, password):
    db = load_db()

    for user in db["users"]:
        if user["username"] == username and user["password"] == hash_password(password):
            return True

    return False

def save_prediction(username, input_data, prediction):
    db = load_db()

    db["predictions"].append({
        "user": username,
        "input": input_data,
        "prediction": float(prediction)
    })

    save_db(db)