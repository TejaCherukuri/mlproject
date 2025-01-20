import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_file_path: str = os.path.join("artifacts", 'train_data.csv')
    test_file_path: str = os.path.join("artifacts", 'test_data.csv')
    raw_file_path: str = os.path.join("artifacts", 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiated data ingestion")
        try:
            # Load the data from csv
            logging.info("Load the data from csv")
            df = pd.read_csv("notebook/data/student_performance.csv")
            
            # Create artifacts folder
            logging.info("creating artifacts folder")
            os.makedirs(os.path.dirname(self.ingestion_config.train_file_path), exist_ok=True)

            # Store the raw data
            logging.info("Storing the raw data")
            df.to_csv(self.ingestion_config.raw_file_path, index=False, header=True)

            # Split the data
            logging.info("Initiated Train-Test Split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Store the train and test data
            logging.info("Storing the training and testing data")
            train_set.to_csv(self.ingestion_config.train_file_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_file_path, index=False, header=True)

            logging.info("Data Ingestion completed")

            return (
                self.ingestion_config.train_file_path,
                self.ingestion_config.test_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)