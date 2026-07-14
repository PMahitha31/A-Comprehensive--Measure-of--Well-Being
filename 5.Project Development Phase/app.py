"""
app.py
----------------------------------
Flask web application that serves the HDI Predictor.
Loads a pre-trained Linear Regression model and exposes:
    - GET  /         : Input form page
    - POST /predict  : Runs prediction and renders the results page
"""

import os
import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "hdi_model.pkl"
FEATURES = ["Life_Expectancy", "Mean_Years_Schooling", "GNI_Per_Capita"]

# --------------------------------------------------------------------------
# Load the trained model at startup
# --------------------------------------------------------------------------
model = None
model_load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("[INFO] HDI prediction model loaded successfully.")
except FileNotFoundError:
    model_load_error = (
        "Model file 'hdi_model.pkl' not found. "
        "Please run 'python train_model.py' first."
    )
    print(f"[WARNING] {model_load_error}")


def categorize_hdi(score: float) -> dict:
    """
    Categorize a numeric HDI score into standard UNDP development tiers.

    Returns a dict containing the category label and a CSS class name
    used to style the result badge.
    """
    if score >= 0.800:
        return {"label": "Very High Human Development", "css_class": "badge-very-high"}
    elif score >= 0.700:
        return {"label": "High Human Development", "css_class": "badge-high"}
    elif score >= 0.550:
        return {"label": "Medium Human Development", "css_class": "badge-medium"}
    else:
        return {"label": "Low Human Development", "css_class": "badge-low"}


def validate_inputs(form) -> tuple:
    """
    Validate and parse the incoming form data.
    Returns (values_dict, errors_list).
    """
    errors = []
    values = {}

    field_rules = {
        "life_expectancy": ("Life Expectancy", 0, 100),
        "mean_years_schooling": ("Mean Years of Schooling", 0, 25),
        "gni_per_capita": ("GNI Per Capita", 0, 200000),
    }

    for field_key, (display_name, min_val, max_val) in field_rules.items():
        raw_value = form.get(field_key, "").strip()

        if not raw_value:
            errors.append(f"{display_name} is required.")
            continue

        try:
            numeric_value = float(raw_value)
        except ValueError:
            errors.append(f"{display_name} must be a valid number.")
            continue

        if numeric_value < min_val or numeric_value > max_val:
            errors.append(
                f"{display_name} must be between {min_val} and {max_val}."
            )
            continue

        values[field_key] = numeric_value

    return values, errors


@app.route("/")
def index():
    """Render the home page with the input form."""
    return render_template("index.html", model_load_error=model_load_error)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission, validate inputs, run prediction, show results."""
    if model is None:
        return render_template(
            "index.html",
            model_load_error=model_load_error or "Model is not available.",
        )

    values, errors = validate_inputs(request.form)

    if errors:
        return render_template("index.html", errors=errors, model_load_error=None)

    # Build feature vector in the exact order the model was trained on
    input_array = np.array([[
        values["life_expectancy"],
        values["mean_years_schooling"],
        values["gni_per_capita"],
    ]])

    raw_prediction = model.predict(input_array)[0]

    # Clip prediction to a realistic HDI range [0, 1]
    clipped_score = float(np.clip(raw_prediction, 0.0, 1.0))
    rounded_score = round(clipped_score, 3)

    category = categorize_hdi(clipped_score)

    return render_template(
        "result.html",
        hdi_score=rounded_score,
        category_label=category["label"],
        category_css=category["css_class"],
        life_expectancy=values["life_expectancy"],
        mean_years_schooling=values["mean_years_schooling"],
        gni_per_capita=values["gni_per_capita"],
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
