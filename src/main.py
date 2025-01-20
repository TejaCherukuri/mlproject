from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

if __name__ == "__main__":
    obj = DataIngestion()
    train_file, test_file = obj.initiate_data_ingestion()

    data_transform_obj = DataTransformation()
    train_arr, test_arr, _ = data_transform_obj.initiate_data_transformation(train_file, test_file)

    model_trainer_obj = ModelTrainer()
    r2_score = model_trainer_obj.initiate_model_trainer(train_arr, test_arr)
    logging.info(f"R2 Score for Student Performance Dataset: {r2_score}")


