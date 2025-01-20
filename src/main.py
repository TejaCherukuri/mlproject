from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    obj = DataIngestion()
    train_file, test_file = obj.initiate_data_ingestion()

    data_transform_obj = DataTransformation()
    data_transform_obj.initiate_data_transformation(train_file, test_file)
