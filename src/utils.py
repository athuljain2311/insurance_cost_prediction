import dill
import sys
import os

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        logging.info('Dumping Preprocessor Object')
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e,sys)

def evaluate_models(x_train,y_train,x_test,y_test,models,param):
    try:
        output = dict()

        for model_name,model in models.items():
            grid = GridSearchCV(model,param_grid=param[model_name],scoring='r2',n_jobs=-1)
            grid.fit(x_train,y_train)
            model.set_params(**grid.best_params_)
            y_test_pred = model.predict(x_test)
            output[model_name] = (r2_score(y_test,y_test_pred),model)
        return output
    
    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e,sys)