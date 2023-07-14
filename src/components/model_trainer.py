import sys
import os

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Initiating Train Test Split')
            x_train,x_test,y_train,y_test = train_arr[:,:-1],test_arr[:,:-1],train_arr[:,-1],test_arr[:,-1]
            
            models = {"LinearRegression":LinearRegression(),
                        "Ridge":Ridge(),
                        "Lasso":Lasso(),
                        "KNeighborsRegressor":KNeighborsRegressor(),
                        "DecisionTreeRegressor":DecisionTreeRegressor(),
                        "RandomForestRegressor":RandomForestRegressor(),
                        "XGBRegressor":XGBRegressor()}
            
            params = {'LinearRegression':{},
                        'Ridge':{'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]},
                        'Lasso':{'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]},
                        'KNeighborsRegressor':{'n_neighbors':[4,5,6,7,8,9],'weights':['uniform','distance']},
                        'DecisionTreeRegressor':{'max_depth':[1,3,5,7,9,11,12],'max_features':["log2","sqrt",None]},
                        'RandomForestRegressor':{'n_estimators':[50,80,100,200],'max_depth':[1,3,5,7,9,11,12],'max_features':["log2","sqrt",None]},
                        'XGBRegressor':{'eta':[0.01,0.05,0.1,0.15,0.2],'max_depth':[1,3,5,7,9]}}
            
            model_report = evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)
            
            best = sorted(model_report.items(),key=lambda x:x[1][0])[-1]
            best_model_name = best[0]
            best_model_score = best[1][0]
            best_model = best[1][1]
            
            if best_model_score<0.6:
                raise CustomException('No Best Model Found',sys)
            logging.info(f"Best Model : {best_model_name}, Score : {best_model_score}")

            #best_model = models[best_model_name](eta=best_model_params['eta'], max_depth=best_model_params['max_depth'])

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            y_test_pred = best_model.predict(x_test)
            return r2_score(y_test,y_test_pred)

        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)