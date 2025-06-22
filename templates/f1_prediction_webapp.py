from flask import Flask, render_template, request
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model and preprocessor
MODEL_PATH = "C:/CODING/f1_webapp/templates/f1_top3_model.keras"
PREPROCESSOR_PATH = "C:/CODING/f1_webapp/templates/preprocessor.joblib"
model = tf.keras.models.load_model(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Dictionary of hardcoded qualifying data for each GP (only examples shown)
qualifying_data = {
    "australia": [
        {"quali_pos": 1, "driver": "Lando Norris", "constructor": "McLaren"},
        {"quali_pos": 2, "driver": "Oscar Piastri", "constructor": "McLaren"},
        {"quali_pos": 3, "driver": "Max Verstappen", "constructor": "Red Bull Racing"},
        {"quali_pos": 4, "driver": "George Russell", "constructor": "Mercedes"},
        {"quali_pos": 5, "driver": "Yuki Tsunoda", "constructor": "Racing Bulls"},
        {"quali_pos": 6, "driver": "Alexander Albon", "constructor": "Williams"},
        {"quali_pos": 7, "driver": "Charles Leclerc", "constructor": "Ferrari"},
        {"quali_pos": 8, "driver": "Lewis Hamilton", "constructor": "Ferrari"},
        {"quali_pos": 9, "driver": "Pierre Gasly", "constructor": "Alpine"},
        {"quali_pos": 10, "driver": "Carlos Sainz", "constructor": "Williams"},
        # Add rest of drivers...
    ],
    # Add more countries with similar structure (e.g. "china": [...])
}

# Function to preprocess and predict top 3
def predict_top3(gp_name):
    gp_key = gp_name.strip().lower()
    if gp_key not in qualifying_data:
        return None, []

    data = qualifying_data[gp_key]
    df = pd.DataFrame(data)

    # Add placeholder or default feature values
    df["GP_name"] = gp_name.title()
    df["driver_confidence"] = 0.9
    df["constructor_reliability"] = 0.9
    df["active_driver"] = 1
    df["active_constructor"] = 1

    # Preprocess
    X_proc = preprocessor.transform(df)
    preds = model.predict(X_proc).flatten()
    df["top3_probability"] = preds
    df_sorted = df.sort_values("top3_probability", ascending=False).head(3)
    return gp_name.title(), df_sorted

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    gp_name = request.form["gp_name"]
    gp, table = predict_top3(gp_name)
    if not table:
        return render_template("results.html", gp=gp_name.title(), table=[], message="❌ No data for this GP. Try another or wait for it.")
    return render_template("results.html", gp=gp, table=table.to_dict(orient="records"), message=None)

if __name__ == "__main__":
    app.run(debug=True)
