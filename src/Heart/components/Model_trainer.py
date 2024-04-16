import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass

from src.Heart.logger import logging
from src.Heart.exception import CustomException
from src.Heart.utils.utils import save_object,evaluate_model
from src.Heart.components.Model_evaluation import eval_metrics



@dataclass
class ModelTrainerConfig:
    model_trainer_config_path=os.path.join("Artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting dependent and independing variable from train and test data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'Logistic Regression':LogisticRegression(),
                'Naive Bayes':GaussianNB(),
                'Random Forest Classfier':RandomForestClassifier(n_estimators=20, random_state=12,max_depth=5),
                'XG Boost':XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,
                                         seed=27, reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5),
                'K Nearest Neighbors':KNeighborsClassifier(n_neighbors=10),
                'Decision Tree':DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6),
                'Support Vector Machine':SVC(kernel='rbf', C=2)
                }
            model_report=evaluate_model(X_train,y_train,X_test,y_test,models)  #Return dict fomat

            logging.info(f"Model Report:{model_report}")

            #To get the best model score from the dictionary
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            save_object(file_path=self.model_trainer_config.model_trainer_config_path,obj=best_model)

            logging.info("Evaluation metric started")
            predict=best_model.predict(X_test)
            accuracy,precision,recall,f1=eval_metrics(y_test,predict)
            logging.info(f"Best Model Found ,Model name :{best_model_name},Accuracy Score:{best_model_score}")
            logging.info("*"*50)
            logging.info(f"Accuracy score is:{accuracy}")
            logging.info(f"precision score is:{precision}")
            logging.info(f"recall score is:{recall}")
            logging.info(f"f1 score is:{f1}")    
            logging.info("*"*50)

            logging.info("Evaluation metrics done...")

            print("DONE.....")
        except Exception as e:
            logging.info("Exception occured at Model Training")
            raise CustomException(e,sys)



