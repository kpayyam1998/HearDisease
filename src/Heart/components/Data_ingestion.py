import os
import sys
import numpy as np
import pandas as pd
from src.Heart.exception import CustomException
from src.Heart.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

from src.Heart.components.Data_transformation import DataTransformation,DataTransformationConfig
from src.Heart.components.Model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("Artifacts","rawdata.csv")
    train_data_path:str=os.path.join("Artifacts","train_data.csv")
    test_data_path:str=os.path.join("Artifacts","test_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Dataingestion is started")

        try:
            data=pd.read_csv("../../../Notebook_Experiments/data/heart.csv")
            logging.info("Data loaded")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Created the raw data file")

            logging.info("Started to split train and test data")
            train_data,test_data=train_test_split(data,test_size=0.2,random_state=42)

            logging.info("Data Splitting is done")

            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Data Splitted successfully")

            return (
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
        obj=DataIngestion()
        train_path,test_path=obj.initiate_data_ingestion()
        data_transformation_obj=DataTransformation()
        train_arr,test_arr=data_transformation_obj.initialize_data_transformation(train_path,test_path)
        model_trainer=ModelTrainer()
        model_trainer.initiate_model_training(train_array=train_arr,test_array=test_arr)


