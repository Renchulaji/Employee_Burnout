from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the entire pipeline (preprocessing + model)
pipeline = joblib.load("burnout_pipeline.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get raw inputs from form
        gender = request.form["gender"]
        company_type = request.form["company_type"]
        wfh_setup = request.form["wfh_setup"]
        mental_fatigue = float(request.form["mental_fatigue"])
        resource_allocation = float(request.form["resource_allocation"])
        designation = float(request.form["designation"])

        # Create DataFrame with same column names as training data (raw values)
        input_df = pd.DataFrame([[
            gender,
            company_type,
            wfh_setup,
            mental_fatigue,
            resource_allocation,
            designation
        ]], columns=[
            "Gender", "Company Type", "WFH Setup Available",
            "Mental Fatigue Score", "Resource Allocation", "Designation"
        ])

        # Predict directly using pipeline
        prediction = pipeline.predict(input_df)[0]

        # Generate suggestion text based on prediction value
        if prediction < 0.3:
            suggestion = "Low burnout risk — maintain current work-life balance."
        elif prediction < 0.6:
            suggestion = "Moderate burnout risk — consider reducing workload or taking breaks."
        else:
            suggestion = "High burnout risk — immediate intervention recommended."

        return render_template(
            "index.html",
            prediction_text=f"Predicted Burn Rate: {prediction:.2f}",
            suggestion_text=suggestion
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)