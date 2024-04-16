import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from src.Heart.exception import CustomException
from src.Heart.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.Heart.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("Artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation(self):
        try:
            logging.info("Data transformation initiated")
            #our data set all the col have numerical col
            numerical_col=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

            num_pipeline=Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy='median')), # handling missing values
                    ("scalar",StandardScaler())
                ]
            )
            preprocessor=ColumnTransformer([
                ("numerical",num_pipeline,numerical_col)
            ])

            return preprocessor


        except Exception as e:
            logging.info("Exception is occured in the initiate data transformation")
            raise CustomException(e,sys)

    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test df is successfully completed")

            preprocessing_obj=self.get_data_transformation()
            
            target_col_name="target"
            drop_col_name=[target_col_name]

            input_feature_traindf=train_df.drop(columns=drop_col_name,axis=1)
            target_feature_train_df=train_df[target_col_name]
            
            input_feature_testdf=test_df.drop(columns=drop_col_name,axis=1)
            target_feature_test_df=test_df[target_col_name]

            logging.info("Splitting train and test feature is completed")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_traindf)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_testdf)

            logging.info("Applying preprocessing object on training and testing datasets")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)] # pasing features and releavent values
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("preprocessing pickle file saved")
            return train_arr,test_arr                      
        except Exception as e:
            raise CustomException(e,sys)