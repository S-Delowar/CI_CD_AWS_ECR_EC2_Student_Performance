import os, sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


@dataclass 
class DataTransformationConfig:
    data_preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_transformation_obj(self):
        try:
            num_columns = ['writing_score', 'reading_score']
            cat_columns = ["gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"]
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f'Preprocessor Pipeline Initiated')
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_columns),
                    ('cat_pipeline', cat_pipeline, cat_columns)
                ]
            )
            logging.info(f'Preprocessor object Created')
            return preprocessor
            
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Data Transformation Starts')
            
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            target_column = "math_score"
            
            input_feature_train_df = train_data.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_data[target_column]
            
            input_feature_test_df = test_data.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_data[target_column]
            
            preprocessor = self.get_transformation_obj()
            logging.info(f'Got the preprocessor object')
            # print(preprocessor)
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f'Saving Preprocessor Object')
            
            save_object(self.data_transformation_config.data_preprocessor_path, preprocessor)
            logging.info(f'Saved Preprocessor Object')
            
            logging.info(f'Data Transformation Completed')
            
            return(
                train_arr, test_arr, self.data_transformation_config.data_preprocessor_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)