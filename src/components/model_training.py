import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "AdaBoost": AdaBoostClassifier(),
            }

            models = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100, 300, 500, 700],
                    'max_depth': [None, 5, 8, 12],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'class_weight': ['balanced', 'balanced_subsample', None]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6],
                    'subsample': [0.6, 0.8, 1.0],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'algorithm': ['SAMME', 'SAMME.R']
                },
                "XGBoost": {
                    'n_estimators': [100, 300, 500, 700],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6, 8],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.2, 0.5],
                    'reg_alpha': [0, 0.01, 0.1],
                    'reg_lambda': [1, 1.5, 2]
                }
            }

            # Evaluate models using classification metrics (RandomizedSearchCV limited to 10 iterations)
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
                n_iter=20  # Pass n_iter to limit search iterations
            )

            # Get best model based on f1-score or accuracy
            best_model_name = max(model_report, key=lambda x: model_report[x]['f1'])
            best_model = models[best_model_name]
            best_score = model_report[best_model_name]['f1']

            if best_score < 0.6:
                raise CustomException("No best model found with acceptable score")

            logging.info(f"Best model found: {best_model_name} with F1-score: {best_score}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict on test set
            y_pred = best_model.predict(X_test)

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred)
            }

            return self.model_trainer_config.trained_model_file_path, best_model_name, best_score


        except Exception as e:
            raise CustomException(e, sys)
