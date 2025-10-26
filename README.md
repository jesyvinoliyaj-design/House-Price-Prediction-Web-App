# House Price Prediction Web App

A web application to predict house prices in India using machine learning techniques and a simple, interactive interface.

## Overview

This project leverages key Python libraries such as **NumPy**, **Pandas**, **Matplotlib**, and **Scikit-learn** for data processing, visualisation, and model training. The web interface is built with **Flask**, enabling users to input house features and obtain predicted prices instantly.

## Features

- **Interactive Web App**: User-friendly interface for quick predictions.
- **End-to-End ML Pipeline**: Data ingestion, preprocessing, training, evaluation, and prediction.
- **Jupyter Notebook Integration**: Exploration & visualization of data and model.
- **Pre-trained Model**: Deployed for fast inference.
- **Scalable & Extendable**: Easily adapt for other regions or features.

## Demo

Access the live application here:  
[house-price-prediction-web-app-b2bq.onrender.com](https://house-price-prediction-web-app-b2bq.onrender.com/)

## Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Flask
- HTML & CSS

## Project Structure

```
static/                   # Static assets (CSS, images, etc.)
templates/                # HTML templates for Flask
House Price Prediction Web App.ipynb   # Notebook: data exploration, modeling
House_Price_India_data.csv             # Dataset
app.py                                 # Flask application entry point
house_price_model.pkl                  # Trained machine learning model
model_feature_columns.pkl              # Model feature column definitions
requirements.txt                       # List of dependencies
```

## Setup Instructions

### Requirements

- Python 3.x
- See `requirements.txt` for required packages

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/jesyvinoliyaj-design/House-Price-Prediction-Web-App.git
   cd House-Price-Prediction-Web-App
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web app:**

   ```bash
   python app.py
   ```

4. **Access via browser:**  
   Visit `http://127.0.0.1:5000`

## File Descriptions

- `House Price Prediction Web App.ipynb` - Jupyter notebook for exploratory analysis, feature engineering, and model training.
- `House_Price_India_data.csv` - Dataset with features and target price.
- `app.py` - Flask app serving the web UI and prediction logic.
- `house_price_model.pkl` - Serialized ML model for prediction.
- `model_feature_columns.pkl` - Model features for preprocessing input.
- `requirements.txt` - Python package dependencies.

## Contributing

Feel free to fork, report issues, and submit pull requests to enhance the project.

## License

This repository has no specific license. Please contact the author if you wish to use it for commercial purposes.

***

**Author:** [jesyvinoliyaj-design](https://github.com/jesyvinoliyaj-design)

***

Let me know if you need any section expanded or customized!

[1](https://github.com/jesyvinoliyaj-design/House-Price-Prediction-Web-App)
