from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collect form data
            HR = request.form.get('HR')
            P = request.form.get('P')
            PR = request.form.get('PR')
            QRS = request.form.get('QRS')
            QT = request.form.get('QT')
            QTc = request.form.get('QTc')
            P_Axis = request.form.get('P_Axis')
            QRS_Axis = request.form.get('QRS_Axis')
            T_Axis = request.form.get('T_Axis')
            RV5 = request.form.get('RV5')
            SV1 = request.form.get('SV1')

            # Validate that all fields are filled
            if not all([HR, P, PR, QRS, QT, QTc, P_Axis, QRS_Axis, T_Axis, RV5, SV1]):
                return render_template('home.html', error="Please fill all input fields.")

            # Convert inputs to float and create CustomData object
            data = CustomData(
                HR=float(HR),
                P=float(P),
                PR=float(PR),
                QRS=float(QRS),
                QT=float(QT),
                QTc=float(QTc),
                P_Axis=float(P_Axis),
                QRS_Axis=float(QRS_Axis),
                T_Axis=float(T_Axis),
                RV5=float(RV5),
                SV1=float(SV1)
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()

            # Run prediction
            pipeline = PredictPipeline()
            results = pipeline.predict(pred_df)

            # Map prediction to label
            label = "No Heart Attack Risk" if results[0] == 0 else "Heart Attack Risk"

            return render_template(
                'home.html',
                results=label,
                prediction=True,
                **request.form
            )

        except Exception as e:
            return render_template('home.html', error=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
