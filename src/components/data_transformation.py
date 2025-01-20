import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    
    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        '''
        This function is responsible for data transformation
        '''
        try:
            logging.info("Defining the numerical and categorical pipelines")
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("onehotencoder", OneHotEncoder())
                ]
            )

            logging.info("Defining the preprocessor object based on ColumnTransformer")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            logging.info("Returning the preprocessor object")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def initiate_data_transformation(self, train_file_path, test_file_path):
        try:

            logging.info("Data Transformation Initiated")
            logging.info("Loading train data and test data")
            train_data = pd.read_csv(train_file_path)
            test_data = pd.read_csv(test_file_path)

            # Define the target column
            target_column = "math_score"

            # Splitting train data into X and y
            logging.info("Defining X_train, y_train and X_test, y_test")
            X_train = train_data.drop(columns=[target_column], axis=1)
            y_train = train_data[target_column]

            X_test = test_data.drop(columns=[target_column], axis=1)
            y_test = test_data[target_column]

            # Figure out the numerical and categorical columns
            num_columns = [feature for feature in X_train.columns if X_train[feature].dtype != 'O']
            cat_columns = [feature for feature in X_train.columns if X_train[feature].dtype == 'O']

            logging.info(f"Numerical Columns: {num_columns}")
            logging.info(f"Categorical Columns: {cat_columns}")

            # Get the preprocessor object
            preprocessor = self.get_data_transformer_object(num_columns, cat_columns)

            # Transform on train and test data
            logging.info("Performing data transformation through preprocessor object using 'fit_transform'")
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            logging.info(f"Shape of X_train_arr: {X_train_arr.shape}")
            logging.info(f"Shape of y_train: {y_train.shape}")

            # Concatenate two arrays
            logging.info("Creating a consolidated numpy array by concatenating X_train and y_train column-wise")
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            logging.info("Creating a consolidated numpy array by concatenating X_test and y_test column-wise")
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            # Save the preprocessor object
            logging.info("Saving the preprocessor/transformer object as pickle file")
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
