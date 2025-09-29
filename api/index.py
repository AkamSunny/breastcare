import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request

app = Flask(__name__)


# Home route
@app.route("/")
def home():
    return render_template("home.html", query="")


# About page
@app.route("/about")
def about():
    return render_template("about.html")


# Contact page
@app.route("/contact")
def contact():
    return render_template("contact.html")


# Prediction route
@app.route("/predict", methods=['POST'])
def cancerPrediction():
    try:
        df_xgb = pd.read_csv("breast_cancer.csv")  # ✅ Path fixed (relative to project root)

        # Get and validate inputs
        inputQuery1 = float(request.form['query1'])
        inputQuery2 = float(request.form['query2'])
        inputQuery3 = float(request.form['query3'])
        inputQuery4 = float(request.form['query4'])
        inputQuery5 = float(request.form['query5'])

        # Select features
        features = [
            'perimeter_worst',
            'concave points_worst',
            'concave points_mean',
            'area_mean',
            'area_worst',
            'diagnosis'
        ]
        xgb_df = df_xgb[features]
        xgb_df['diagnosis'] = xgb_df['diagnosis'].map({'M': 1, 'B': 0})

        X = xgb_df.drop('diagnosis', axis=1)
        y = xgb_df['diagnosis']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # XGBoost model
        model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            scale_pos_weight=357/212,
            reg_alpha=0.5,
            reg_lambda=2.0,
            subsample=0.7,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Prepare new input
        new_data = pd.DataFrame([[
            inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5
        ]], columns=['perimeter_worst', 'concave points_worst', 'concave points_mean', 'area_mean', 'area_worst'])

        new_data_scaled = scaler.transform(new_data)

        # Prediction
        prediction = model.predict(new_data_scaled)
        proba = model.predict_proba(new_data_scaled)

        if prediction[0] == 1:
            output1 = "The patient is diagnosed with Breast Cancer"
            output2 = f"Confidence: {proba[0][1] * 100:.2f}%"
        else:
            output1 = "The patient is not diagnosed with Breast Cancer"
            output2 = f"Confidence: {proba[0][0] * 100:.2f}%"

        return render_template(
            "home.html",
            output1=output1,
            output2=output2,
            query1=request.form['query1'],
            query2=request.form['query2'],
            query3=request.form['query3'],
            query4=request.form['query4'],
            query5=request.form['query5']
        )

    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        return render_template(
            "home.html",
            output1=error_message,
            output2="",
            query1=request.form.get('query1', ''),
            query2=request.form.get('query2', ''),
            query3=request.form.get('query3', ''),
            query4=request.form.get('query4', ''),
            query5=request.form.get('query5', '')
        )


# ✅ This allows local development but is ignored on Vercel
if __name__ == "__main__":
    app.run(debug=True, port=5000)
