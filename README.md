# Heart Attack Detection

This project is a machine learning application designed to predict the risk of a heart attack based on various Electrocardiogram (ECG) features and health metrics. It provides a user-friendly web interface for inputting patient data and receiving instant predictions.

## 🚀 Features

*   **Prediction Model:** Utilizes a trained machine learning model to analyze input features and predict heart attack risk.
*   **Web Interface:** A clean and responsive web application built with Flask for easy interaction.
*   **Real-time Analysis:** Get immediate results after submitting the patient's data.
*   **Data Processing:** Includes a robust data pipeline for preprocessing and feature scaling.

## 🛠️ Technology Stack

*   **Python:** Core programming language.
*   **Flask:** Web framework for the application.
*   **Scikit-learn & XGBoost:** Machine learning libraries for model training and prediction.
*   **Pandas & NumPy:** Data manipulation and numerical operations.
*   **HTML/CSS:** Frontend for the web interface.

## 📂 Project Structure

```
├── artifacts/          # Stores trained models and preprocessor objects
├── notebooks/          # Jupyter notebooks for EDA and model training
├── src/                # Source code for the project
│   ├── components/     # Data ingestion, transformation, and model trainer modules
│   ├── pipeline/       # Prediction and training pipelines
│   ├── utils.py        # Utility functions
│   ├── logger.py       # Logging configuration
│   └── exception.py    # Custom exception handling
├── templates/          # HTML templates for the web application
├── app.py              # Main Flask application entry point
├── requirements.txt    # Project dependencies
└── setup.py            # Package setup configuration
```

## 🔧 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abhaykanwar24/HeartAttackDetection.git
    cd HeartAttackDetection
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

1.  **Run the application:**
    ```bash
    python app.py
    ```

2.  **Access the web interface:**
    Open your web browser and go to `http://127.0.0.1:5000/`.

3.  **Make a prediction:**
    Click on the "Predict" button or navigate to the prediction page. Fill in the required ECG parameters (HR, P, PR, QRS, QT, QTc, etc.) and submit the form to see the result.

## 📊 Model Training

The model training process is documented in the `notebooks/` directory. The `EDA_and_feature_engineering.ipynb` notebook covers data analysis, while `model_training.ipynb` details the model selection and training steps.

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License.
