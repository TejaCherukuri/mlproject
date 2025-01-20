import joblib
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, object):
    try:
        logging.info("Dumping the object as pickle file")
        joblib.dump(object, file_path)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, X_test, y_train, y_test, models):
    try:
        r2_list = {}
        for i in range(len(list(models))):

            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            logging.info(f"=====Training on {model_name}======")

            model.fit(X_train, y_train)
            logging.info(f"=====Predicting on {model_name}======")
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            logging.info(f"=====R2 Score for {model_name}====== : {r2}")
            r2_list[model_name] = r2

        return r2_list

    except Exception as e:
        raise CustomException(e, sys)