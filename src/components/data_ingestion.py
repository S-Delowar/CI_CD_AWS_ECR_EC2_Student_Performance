import os, sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_train import ModelTrainerConfig, ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')
    
class DataIngestion():
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion starts")
        try:
            df = pd.read_csv('notebook\data\StudentsPerformance2.csv')
            logging.info('Read data as DataFrame')
            
            train_set, test_set = train_test_split(df, test_size=.20, random_state=42)
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion completed. Saved Train, Test data")
            
            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    
    transformation_obj = DataTransformation()
    train_arr, test_arr, preprocessor_obj = transformation_obj.initiate_data_transformation(train_path, test_path)
    
    # print(f'train_arr_shape from transformation: {train_arr.shape}')
    # print(f'test_arr_shape from transformation: {test_arr.shape}')
    
    model_trainer_obj = ModelTrainer()
    model_trainer_obj.initiate_model_trainer(train_arr, test_arr)
    # print('Everything is okay.')