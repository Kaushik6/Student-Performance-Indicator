import os
import sys
import dill

import numpy as np
import pandas as pd

from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

##CatBoostRegressor is not fully compatible with sklearn’s GridSearchCV because it does not implement sklearn’s internal estimator tags, so hyperparameter tuning must be handled separately.

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report={}

        for model_name, model in models.items():
            para=param.get(model_name,{})

            if model_name == "CatBoost Regressor":
                CB_model=CatBoostRegressor(**para,verbose=False)
                CB_model.fit(X_train,y_train)
                y_train_pred = CB_model.predict(X_train)
                y_test_pred = CB_model.predict(X_test)
                train_model_score = r2_score(y_train,y_train_pred)
                test_model_score = r2_score(y_test,y_test_pred)
                report[model_name] = test_model_score
                continue

            if para:
                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)
                model.set_params(**gs.best_params_)
            
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)

            test_model_score = r2_score(y_test,y_test_pred)

            report[model_name] = test_model_score
        
        return report
    except Exception as e:
        raise CustomException(e,sys)