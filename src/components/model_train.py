import os, sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr):
        logging.info(f'Model Training Starts')
        try:
            # initialize model dictionary
            models_dict ={
                'Linear Regression': LinearRegression(),
                'KNeighborsRegressor':KNeighborsRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'XGBRegressor': XGBRegressor()
            }
            
            # parameters for hyperparameter tuning
            params = {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighborsRegressor":{}          
            }
            
            # Split Train and Test Data
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            
            # print(f'X_train shape:{X_train.shape}, y_train shape: {y_train.shape}, X_test_shape:{X_test.shape}, y_test_shape:{y_test.shape} ')
            
            # get best model, best score and the evaluation report
            best_model, best_score = evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test = y_test, models_dict=models_dict, params=params) 
            
            if best_score < 0.7:
                raise CustomException("No best model found")
            else:
                logging.info(f'Best model- {best_model} with score {round(best_score,2)*100}%')
                logging.info(f'Best Parameters - {best_model.get_params()}')
            
            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )            
            
        except Exception as e:
            raise CustomException(e, sys)
    
    
