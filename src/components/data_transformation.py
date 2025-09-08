import sys
from dataclasses import dataclass

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts' , "preprocessor.pkl")


class DataTransformation:
    '''
    this fucntion is responsible  for data tranformation based of different types of data
    '''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['HR','P','PR','QRS','QT','QTc','P_Axis','QRS_Axis','T_Axis','RV5','SV1']

            num_pipeline = Pipeline(
                steps = [
                ("imputer",SimpleImputer(strategy='mean')),
                ("scaler" , StandardScaler())
                ]  
            )
            
            logging.info(f"Numerical columns: {numerical_columns}")
                

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline" , num_pipeline , numerical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train and test CSVs, applies preprocessing transformations,
        and returns transformed arrays along with the preprocessor path.
        """
        try:
            # Read datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Train and test data read successfully. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # Get preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()  # Ensure this method exists in the class

            target_column_name = "target"

            # Separate input features and target
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data")

            # Transform features
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            # Combine features and target into arrays
            train_arr = np.c_[X_train_arr, y_train.to_numpy()]
            test_arr = np.c_[X_test_arr, y_test.to_numpy()]

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Preprocessing object saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
