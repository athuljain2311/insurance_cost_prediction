import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException

from data_transformation import DataTransformation
from model_trainer import ModelTrainer

from dataclasses import dataclass

sys.path.append("D:\Projects\FullStack_Projects\insurance_cost_prediction\src")

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts','data.csv')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Started')
        try:
            df = pd.read_csv('data\insurance.csv')
            logging.info('Succesfully Read the Data')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Initiating Train-Test Split')
            train_df,test_df = train_test_split(df,test_size=0.2,random_state=23)

            train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion Completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_arr,test_arr = data_transformation_obj.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer_obj = ModelTrainer()
    logging.info(f"Score : {model_trainer_obj.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr)}")