import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path , "wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param, n_iter=None):
    """
    Evaluate multiple classification models with optional RandomizedSearchCV.
    """
    report = {}

    for model_name, model in models.items():
        params_grid = param.get(model_name, {})

        if n_iter:  # Use RandomizedSearchCV if n_iter is provided
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params_grid,
                n_iter=n_iter,
                cv=3,
                n_jobs=-1,
                scoring='f1',  # Use F1-score for classification
                random_state=42,
                refit=True
            )
        else:  # Fall back to full GridSearchCV
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(
                estimator=model,
                param_grid=params_grid,
                cv=3,
                n_jobs=-1,
                scoring='f1',
                refit=True
            )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        report[model_name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        }

        # Update original model to the best found
        models[model_name] = best_model

    return report

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)