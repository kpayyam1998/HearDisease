import os
import sys
# import mlflow
# import pickle
# import numpy as np
# import pandas as pd
# import mlflow.sklearn
# from urllib.parse import urlparse
# from src.Heart.utils.utils import load_object
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from src.Heart.exception import CustomException
from src.Heart.logger import logging

def eval_metrics(actual,pred):
    try:
        accuracy=accuracy_score(actual,pred)
        precision=precision_score(actual,pred)
        recall=recall_score(actual,pred)
        f1=f1_score(actual,pred)
        return accuracy,precision,recall,f1
    except Exception as e:
        logging.info("Exception occured in the Model evaluation time")
        raise CustomException(e,sys)

    
    

