import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from src.entity.hf_estimator import HFModelEstimator
from src.exception import MyException
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.logger import logging
from src.utils.main_utils import load_object , read_yaml_file


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[HFModelEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.

        Output      :   Returns model object if available in Hugging Face Hub repo
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            repo_id = self.model_eval_config.hf_repo_id
            model_path = self.model_eval_config.hf_model_path
            hf_estimator = HFModelEstimator(repo_id=repo_id, model_path=model_path)

            if hf_estimator.is_model_present(model_path=model_path):
                return hf_estimator
            return None
        except Exception as e:
            raise MyException(e, sys)

    def _drop_id_column(self, df):
        """Drop LoanID if it exists — matches DataTransformation."""
        logging.info("Dropping id column")
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
        return df

    def _map_binary_columns(self, df):
        """Map No/Yes columns to 0/1 — matches DataTransformation."""
        logging.info("Mapping binary columns to 0/1")
        for col in self._schema_config['binary_columns']:
            df[col] = df[col].map({"No": 0, "Yes": 1})
        return df

    def _map_ordinal_columns(self, df):
        """Map Education to an ordinal scale — matches DataTransformation."""
        logging.info("Mapping Education to ordinal scale")
        education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
        df["Education"] = df["Education"].map(education_map)
        return df

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model
                        with production model and choose best model

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(columns=[TARGET_COLUMN]), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")

            x = self._drop_id_column(x)
            x = self._map_binary_columns(x)
            x = self._map_ordinal_columns(x)

            training_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded ...")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info("Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(
                    f"F1_Score-Production Model: {best_model_f1_score}, "
                    f"F1_Score-New Trained Model: {trained_model_f1_score}"
                )

            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference=trained_model_f1_score - tmp_best_model_score
            )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation

        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            hf_model_path = self.model_eval_config.hf_model_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                hf_model_path=hf_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e