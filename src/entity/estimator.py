import sys
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging


class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object, threshold: float = 0.5):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model
        :param threshold: Locked probability cutoff (from notebook tuning), used instead
                           of the model's default 0.5 cutoff
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self.threshold = threshold

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Function accepts preprocessed inputs (with all custom transformations already applied),
        applies scaling using preprocessing_object, and performs prediction on transformed features
        using the locked probability threshold instead of the model's default 0.5 cutoff.
        """
        try:
            logging.info("Starting prediction process.")

            transformed_feature = self.preprocessing_object.transform(dataframe)
            logging.info(f"Using the trained model to get predictions at threshold={self.threshold:.4f}")
            probabilities = self.trained_model_object.predict_proba(transformed_feature)[:, 1]
            predictions = (probabilities >= self.threshold).astype(int)
            return predictions
        
        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}(threshold={self.threshold})"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}(threshold={self.threshold})"