import os
import sys

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

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
            logging.info('Reading Train and Test Data')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read Train and Test Data')

            logging.info('Obtaining Preprocessor Object')
            preprocessor_obj = self.get_data_transformer_object()
            logging.info('Obtained Preprocessor Object')

            target_column = 'charges'

            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessor_obj
            )

            return (train_arr,test_arr)
        
        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)