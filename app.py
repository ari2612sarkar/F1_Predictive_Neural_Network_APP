from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
from keras.models import load_model
import os

app = Flask(__name__)

# Paths
#MODEL_PATH = r"C:\CODING\f1_webapp\templates\f1_top3_model.keras"
#PREPROCESSOR_PATH = r"C:\CODING\f1_webapp\templates\preprocessor.joblib"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "templates", "f1_top3_model.keras")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "templates", "preprocessor.joblib")

# Load model and preprocessor
model = load_model(MODEL_PATH)
if model is None:
    raise ValueError("Model not loaded properly.")

preprocessor = joblib.load(PREPROCESSOR_PATH)

# Serial number → (GP Name, CSV File Path)
'''gp_serial_map = {
    "1": ("Australia", r"C:\CODING\f1_webapp\2025 GP\2025_qualifying_australia.csv"),
    "2": ("China", r"C:\CODING\f1_webapp\2025 GP\2025_qualifying_china.csv"),
    "3": ("Japan", r"C:\CODING\f1_webapp\2025 GP\2025_qualifying_japan.csv"),
    "4": ("Bahrain", r"C:\CODING\f1_webapp\2025 GP\2025_qualifying_bahrain.csv"),
    "5": ("Miami", r"C:\CODING\f1_webapp\2025 GP\2025_qualifying_miami.csv"),
    "6": ("Emilia Romagna", r"C:\CODING\f1_webapp\2025 GP\2025_qualifying_emilia_romagna.csv"),
    "7": ("Monaco", r"C:\CODING\f1_webapp\2025 GP\2025_qualifying_monaco.csv"),
    "8": ("Spain", r"C:\CODING\f1_webapp\2025 GP\2025_qualifying_spain.csv")
}'''

gp_serial_map = {
    "1": ("Australia", os.path.join(BASE_DIR, "2025 qualify GP", "2025_qualifying_australia.csv")),
    "2": ("China", os.path.join(BASE_DIR, "2025 qualify GP", "2025_qualifying_china.csv")),
    "3": ("Japan", os.path.join(BASE_DIR, "2025 qualify GP", "2025_qualifying_japan.csv")),
    "4": ("Bahrain", os.path.join(BASE_DIR, "2025 qualify GP", "2025_qualifying_bahrain.csv")),
    "5": ("Miami", os.path.join(BASE_DIR, "2025 qualify GP", "2025_qualifying_miami.csv")),
    "6": ("Emilia Romagna", os.path.join(BASE_DIR, "2025 qualify GP", "2025_qualifying_emilia_romagna.csv")),
    "7": ("Monaco", os.path.join(BASE_DIR, "2025 qualify GP", "2025_qualifying_monaco.csv")),
    "8": ("Spain", os.path.join(BASE_DIR, "2025 qualify GP", "2025_qualifying_spain.csv"))
    
}


@app.route("/")
def index():
    return render_template("index.html", gp_serial_map=gp_serial_map)

@app.route("/predict", methods=["POST"])
def predict():
    serial = request.form.get("gp_serial")
    if serial not in gp_serial_map:
        return render_template("results.html", table=[], gp="Unknown")

    gp_name, csv_path = gp_serial_map[serial]

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return render_template("results.html", table=[], gp=gp_name)

    # Add expected features
    df["GP_name"] = gp_name
    df["driver_confidence"] = df["quali_pos"].apply(lambda x: 0.95 if x <= 3 else 0.9)
    df["constructor_reliability"] = 0.9
    df["active_driver"] = 1
    df["active_constructor"] = 1

    # Define expected columns
    expected_features = [
        'GP_name', 'quali_pos', 'constructor', 'driver',
        'driver_confidence', 'constructor_reliability',
        'active_driver', 'active_constructor'
    ]

    try:
        X = df[expected_features]
        X_proc = preprocessor.transform(X)
        probs = model.predict(X_proc.toarray()).flatten()
    except Exception as e:
        return render_template("results.html", table=[], gp=gp_name)

    # Append predictions
    df["top3_probability"] = np.round(probs, 6)
    df["predicted_top3"] = (probs > 0.5).astype(int)

    top3 = df.sort_values(by="top3_probability", ascending=False).head(3)
    results = top3[["quali_pos", "driver", "constructor", "top3_probability"]].to_dict(orient="records")

    return render_template("results.html", table=results, gp=gp_name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

