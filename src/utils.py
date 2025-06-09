import os
import sys
import pickle
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_models = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")

            model_param = params.get(model_name, None)

            if model_param:
                logging.info(f"Tuning hyperparameters for {model_name} using GridSearchCV")
                gs = GridSearchCV(model, model_param, cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            best_models[model_name] = best_model

            logging.info(f"Finished evaluating {model_name}, R2 Score: {test_model_score}")

        return report, best_models

    except Exception as e:
        raise CustomException(e, sys)
