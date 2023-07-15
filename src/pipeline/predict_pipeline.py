import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            processed_data = preprocessor.transform(features)
            return model.predict(processed_data)
        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,age,sex,bmi,children,smoker,region):
        self.age=age
        self.sex=sex
        self.bmi=bmi
        self.children=children
        self.smoker=smoker
        self.region=region

    def get_data_as_dataframe(self):
        try:
            custom_dict = {
                'age':[self.age],
                'sex':[self.sex],
                'bmi':[self.bmi],
                'children':[self.children],
                'smoker':[self.smoker],
                'region':[self.region]
            }
            return pd.DataFrame(custom_dict)
        
        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)