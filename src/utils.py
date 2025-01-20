import joblib
import sys
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, object):
    try:
        logging.info("Dumping the object as pickle file")
        joblib.dump(object, file_path)
    except Exception as e:
        raise CustomException(e, sys)