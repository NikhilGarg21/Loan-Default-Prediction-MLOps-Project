import sys
from typing import Tuple

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from src.entity.estimator import MyModel


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(
        self, train: np.array, test: np.array
    ) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   Trains a CatBoostClassifier with the notebook's tuned hyperparameters
                        and evaluates it at the F1-optimal threshold instead of the default 0.5

        Output      :   Returns trained model object and metric artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training CatBoostClassifier with tuned parameters")

            x_train, y_train, x_test, y_test = (
                train[:, :-1],
                train[:, -1],
                test[:, :-1],
                test[:, -1],
            )
            logging.info("train-test split done.")

            model = CatBoostClassifier(
                iterations=self.model_trainer_config._iterations,
                learning_rate=self.model_trainer_config._learning_rate,
                depth=self.model_trainer_config._depth,
                l2_leaf_reg=self.model_trainer_config._l2_leaf_reg,
                subsample=self.model_trainer_config._subsample,
                random_strength=self.model_trainer_config._random_strength,
                random_state=self.model_trainer_config._random_state,
                verbose=False,
            )

            logging.info("Model training going on...")
            model.fit(x_train, y_train)
            logging.info("Model training done.")

            y_prob = model.predict_proba(x_test)[:, 1]
            y_pred = (y_prob >= self.model_trainer_config._model_threshold).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            logging.info(
                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, F1: {f1:.4f}"
            )
            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall,
                accuracy_score=accuracy,
            )
            return model, metric_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps

        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Starting Model Trainer Component")
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )
            logging.info("train-test data loaded")

            trained_model, metric_artifact = self.get_model_object_and_report(
                train=train_arr, test=test_arr
            )
            logging.info("Model object and artifact loaded.")

            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            logging.info("Preprocessing obj loaded.")

            train_prob = trained_model.predict_proba(train_arr[:, :-1])[:, 1]
            train_pred = (
                train_prob >= self.model_trainer_config._model_threshold
            ).astype(int)
            if (
                accuracy_score(train_arr[:, -1], train_pred)
                < self.model_trainer_config.expected_accuracy
            ):
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")

            logging.info("Saving new model as performance is better than previous one.")
            my_model = MyModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=trained_model,
                threshold=self.model_trainer_config._model_threshold,
            )
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info(
                "Saved final model object that includes preprocessing, model, and tuned threshold"
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e
