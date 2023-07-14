import dill
import sys
import os
from logger import logging
from exception import CustomException

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
