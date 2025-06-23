# 🏎️ F1 2025 Podium Predictor

**Live Demo**: [f1-podium-predictor.onrender.com](https://f1-podium-predictor.onrender.com)

Welcome to the **F1 2025 Podium Predictor** — a machine learning-powered web application that predicts the **top 3 drivers** for each Formula 1 Grand Prix based on qualifying session data.

---

## 🚀 Features

* 🌟 Predict podium finishers using a trained neural network
* 📊 Realistic F1 data processing with scikit-learn + TensorFlow
* 🌍 Simple and intuitive UI built using Flask
* 🎉 Stylish result page with driver/team visuals and confetti
* 🚀 Live-hosted on Render with easy navigation

---

## 🖼️ Demo Screenshot

![App Screenshot](static/Screenshot%202025-06-23%20233721.png)

---

## 🧠 Model Performance

| Metric        | Value                 |
| ------------- | --------------------- |
| Accuracy      | **95.6%**             |
| Precision (0) | 96.6%                 |
| Recall (0)    | 97.9%                 |
| F1-Score (0)  | 97.3%                 |
| Model Type    | Binary Neural Network |
| Library       | TensorFlow/Keras      |

---

## 📆 Folder Structure

```
F1_Predictive_Neural_Network_APP/
├── app.py                      # Main Flask app backend
├── requirements.txt           # Dependency list
├── runtime.txt                # Python version hint
├── templates/                 # HTML templates and model files
│   ├── index.html             # Home page
│   ├── results.html           # Results visualization
│   ├── f1_top3_model.keras    # Trained Keras model
│   └── preprocessor.joblib    # Column transformer (scikit-learn)
├── static/                    # Static assets
│   ├── drivers/               # Driver images
│   ├── team/                  # Team logos
│   ├── wallpaper.jpg          # Background image
│   └── Screenshot ...         # Screenshot for demo
├── 2025 qualify GP/           # GP CSV data files
│   └── *.csv
```

---

## 🛠️ Tech Stack

* **Frontend**: HTML, CSS (Orbitron fonts)
* **Backend**: Python (Flask), Gunicorn
* **ML**: TensorFlow / Keras
* **Data Preprocessing**: scikit-learn ColumnTransformer
* **Deployment**: Render (Free Web Service)

---

## 📃 How to Run Locally

```bash
# Clone the repository
https://github.com/ari2612sarkar/F1_Predictive_Neural_Network_APP.git
cd F1_Predictive_Neural_Network_APP

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🧠 How It Works

1. Select a Grand Prix from dropdown.
2. CSV qualifying data is read (position, constructor, driver).
3. Features are preprocessed and passed to the model.
4. The model predicts each driver's podium probability.
5. Top 3 drivers are shown with animation and visual assets.

---

## 🏑️ Render Deployment Steps

* Runtime version set via `runtime.txt`
* Dependencies loaded using `requirements.txt`
* Start command used: `gunicorn app:app`
* Hosted at: [https://f1-podium-predictor.onrender.com](https://f1-podium-predictor.onrender.com)

---

## 🚀 Future Enhancements

* Compare predictions with actual race results
* Add driver profiles and season stats
* Enhance UI responsiveness and animations
* Incorporate live weather or track data

---

## ✨ Author

Made with passion for ML & F1 by **Aritra Sarkar**.

> For queries, suggestions, or feedback, feel free to connect via GitHub.
