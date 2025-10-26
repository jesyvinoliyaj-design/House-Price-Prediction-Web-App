# ===== 1. Imports & Setup =====
import os
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

MODEL_FILE = "house_price_model.pkl"
FEATURES_FILE = "model_feature_columns.pkl"
DATA_FILE = "House_Price_India_data.csv"

# ===== 2. Load Model & Feature Columns =====
if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError("Missing model or feature column files.")
model = joblib.load(MODEL_FILE)
feature_cols = joblib.load(FEATURES_FILE)

# ===== 3. Default price_per_sqft =====
default_price_per_sqft = 1000.0
if os.path.exists(DATA_FILE):
    try:
        df_data = pd.read_csv(DATA_FILE)
        df_data.columns = df_data.columns.str.lower().str.strip()
        if "price_per_sqft" in df_data.columns:
            default_price_per_sqft = float(df_data["price_per_sqft"].median())
        elif "price" in df_data.columns and "area" in df_data.columns:
            default_price_per_sqft = float((df_data["price"] / df_data["area"]).median())
    except Exception:
        default_price_per_sqft = 1000.0

# ===== 4. Helper: build feature row =====
def build_feature_row(area, bedrooms, age, price_per_sqft, location):
    row = {c: 0 for c in feature_cols}
    if "area" in row:
        row["area"] = float(area)
    if "bedrooms" in row:
        row["bedrooms"] = int(bedrooms)
    if "age" in row:
        row["age"] = int(age)
    if "price_per_sqft" in row:
        row["price_per_sqft"] = float(price_per_sqft)
    if location:
        loc_variants = [
            f"loc_{location}",
            f"loc_{location.lower()}",
            f"loc_{location.lower().replace(' ', '_')}"
        ]
        for loc in loc_variants:
            if loc in row:
                row[loc] = 1
                break
    X = pd.DataFrame([row], columns=feature_cols)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return X

# ===== 5. Main route (form + optional JSON forwarding) =====
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if request.is_json:
            return api_predict()
        form = request.form
        try:
            area = float(form.get("area", 0))
        except:
            area = 0.0
        try:
            bedrooms = int(float(form.get("bedrooms", 0)))
        except:
            bedrooms = 0
        age_val = form.get("age", "").strip()
        built_year_val = form.get("built_year", "").strip()
        age = None
        if age_val:
            try:
                age = int(float(age_val))
            except:
                age = None
        elif built_year_val:
            try:
                built_year = int(float(built_year_val))
                age = 2025 - built_year
            except:
                age = None
        if age is None:
            age = 5
        try:
            price_per_sqft = float(form.get("price_per_sqft", default_price_per_sqft))
        except:
            price_per_sqft = default_price_per_sqft
        location = form.get("location", "").strip()
        X = build_feature_row(area, bedrooms, age, price_per_sqft, location)
        pred = model.predict(X)[0]
        pred_value = round(float(pred), 2)
        formatted = f"₹{pred_value:,.2f}"
        # render same index.html with prediction context
        return render_template("index.html",
                               prediction=pred_value,
                               formatted_prediction=formatted,
                               input_summary={
                                   "area": area,
                                   "bedrooms": bedrooms,
                                   "age": age,
                                   "price_per_sqft": price_per_sqft,
                                   "location": location or "N/A"
                               })
    # GET
    return render_template("index.html", prediction=None, formatted_prediction=None, input_summary=None)

# ===== 6. JSON API endpoint =====
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400
    data = request.get_json()
    try:
        area = float(data.get("area", 0))
    except:
        area = 0.0
    try:
        bedrooms = int(float(data.get("bedrooms", 0)))
    except:
        bedrooms = 0
    age = None
    if "age" in data and str(data.get("age")).strip():
        try:
            age = int(float(data.get("age")))
        except:
            age = None
    elif "built_year" in data and str(data.get("built_year")).strip():
        try:
            built_year = int(float(data.get("built_year")))
            age = 2025 - built_year
        except:
            age = None
    if age is None:
        age = int(data.get("age", 5)) if "age" in data else 5
    try:
        price_per_sqft = float(data.get("price_per_sqft", default_price_per_sqft))
    except:
        price_per_sqft = default_price_per_sqft
    location = data.get("location", "").strip()
    X = build_feature_row(area, bedrooms, age, price_per_sqft, location)
    pred = model.predict(X)[0]
    pred_value = round(float(pred), 2)
    return jsonify({
        "status": "success",
        "input": {"area": area, "bedrooms": bedrooms, "age": age, "price_per_sqft": price_per_sqft, "location": location},
        "predicted_price": pred_value,
        "formatted_price": f"₹{pred_value:,.2f}",
        "unit": "INR",
        "metadata": {"model": type(model).__name__, "features": len(feature_cols)}
    }), 200

# ===== 7. Run server =====
if __name__ == "__main__":
    app.run()
