import os
import sys

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['age','bmi','children']
            categorical_columns = ['sex','smoker','region']

            num_pipeline = Pipeline(
                steps=[('scaler',StandardScaler(with_mean=False))]
                )
            
            cat_pipeline = Pipeline(
                        steps=[('ohe',OneHotEncoder()),
                            ('scaler',StandardScaler(with_mean=False))]
                )
            
            logging.info(f'Numerical Features : {numerical_columns}')
            logging.info(f'Categorical Features : {categorical_columns}')

            preprocessor = ColumnTransformer(
                                [('num_pipeline',num_pipeline,numerical_columns),
                                ('cat_pipeline',cat_pipeline,categorical_columns)]
                            )
            
            return preprocessor

        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            pass
        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)