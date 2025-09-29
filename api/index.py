from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib   # for saving/loading scaler
import os

app = Flask(__name__)

#  Load model
model = XGBClassifier()
model.load_model("xgb_breastcare.json")

# Load saved scaler
scaler = joblib.load("scaler.pkl")

@app.route("/ping")
def ping():
    return {"status": "ok", "message": "Flask app is alive!"}

@app.route("/")
def home():
    return render_template("home.html", query="")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["POST"])
def cancerPrediction():
    try:
        #  Get inputs from form
        inputQuery1 = float(request.form['query1'])
        inputQuery2 = float(request.form['query2'])
        inputQuery3 = float(request.form['query3'])
        inputQuery4 = float(request.form['query4'])
        inputQuery5 = float(request.form['query5'])

        #  Put inputs into a DataFrame
        data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
        new_df = pd.DataFrame(data, columns=[
            'perimeter_worst',
            'concave points_worst',
            'concave points_mean',
            'area_mean',
            'area_worst'
        ])

        #  Scale input features
        new_df_scaled = scaler.transform(new_df)

        #  Prediction
        real_pred = model.predict(new_df_scaled)
        real_proba = model.predict_proba(new_df_scaled)

        if real_pred[0] == 1:
            output1 = "The patient is diagnosed with Breast Cancer"
            confidence = float(real_proba[0][1] * 100)
            output2 = f"Confidence: {confidence:.2f}%"
        else:
            output1 = "The patient is not diagnosed with Breast Cancer"
            confidence = float(real_proba[0][0] * 100)
            output2 = f"Confidence: {confidence:.2f}%"

        return render_template(
            "home.html",
            output1=output1, output2=output2,
            query1=request.form['query1'], query2=request.form['query2'],
            query3=request.form['query3'], query4=request.form['query4'],
            query5=request.form['query5']
        )

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return render_template("home.html", output1=error_message, output2="")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
