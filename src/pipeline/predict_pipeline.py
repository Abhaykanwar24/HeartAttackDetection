import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            # Validate input DataFrame
            expected_columns = ['HR', 'P', 'PR', 'QRS', 'QT', 'QTc', 'P_Axis', 'QRS_Axis', 'T_Axis', 'RV5', 'SV1']
            missing_cols = [col for col in expected_columns if col not in features.columns]
            if missing_cols:
                raise CustomException(f"Input DataFrame missing required columns: {missing_cols}", sys)

            # Load model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Apply preprocessing and predict
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, HR, P, PR, QRS, QT, QTc, P_Axis, QRS_Axis, T_Axis, RV5, SV1):
        self.HR = HR
        self.P = P
        self.PR = PR
        self.QRS = QRS
        self.QT = QT
        self.QTc = QTc
        self.P_Axis = P_Axis
        self.QRS_Axis = QRS_Axis
        self.T_Axis = T_Axis
        self.RV5 = RV5
        self.SV1 = SV1

    def get_data_as_data_frame(self):
        try:
            data_dict = {
                'HR': [self.HR],
                'P': [self.P],
                'PR': [self.PR],
                'QRS': [self.QRS],
                'QT': [self.QT],
                'QTc': [self.QTc],
                'P_Axis': [self.P_Axis],
                'QRS_Axis': [self.QRS_Axis],
                'T_Axis': [self.T_Axis],
                'RV5': [self.RV5],
                'SV1': [self.SV1]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)
