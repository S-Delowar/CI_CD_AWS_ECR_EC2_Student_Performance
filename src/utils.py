import os, sys
import numpy as np
import pandas as pd
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    # create a folder -> open the folder -> dump the object here
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
    

def evaluate_model(X_train, X_test, y_train, y_test, models_dict:dict, params:dict):
    logging.info(f'Evaluation starts')
    
    model_list =[]
    model_score_list = []
    
    for i in range(len(models_dict)):
        model = list(models_dict.values())[i]
        model_key = list(models_dict.keys())[i]
        param = params.get(model_key)
        
        gs = GridSearchCV(model, param, cv=3)
        gs.fit(X_train, y_train)
        
        model.set_params(**gs.best_params_)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_score = r2_score(y_test, y_pred)
        
        model_list.append(model)
        model_score_list.append(model_score)
    
    logging.info(f'Evaluation Ended')
    logging.info(f'Generating Evaluation Report')
    evaluation_report = list(zip(model_list, model_score_list))
    evaluation_report = sorted(evaluation_report, key=lambda x: x[1], reverse=True)
    
    best_model_with_score = evaluation_report[0]
    best_model = best_model_with_score[0]
    best_score = best_model_with_score[1]
    
    logging.info(f'Evaluation Report Generated')
    
    return best_model, best_score