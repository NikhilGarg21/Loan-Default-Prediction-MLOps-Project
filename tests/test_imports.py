"""
Import sanity checks — catches broken imports, typos in class names,
or a missing dependency before a Docker build wastes time on it.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_component_imports():
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer
    from src.components.model_evaluation import ModelEvaluation
    from src.components.model_pusher import ModelPusher
    assert all([DataIngestion, DataValidation, DataTransformation,
                ModelTrainer, ModelEvaluation, ModelPusher])


def test_entity_imports():
    from src.entity.config_entity import (
        DataIngestionConfig, DataValidationConfig, DataTransformationConfig,
        ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig,
    )
    from src.entity.artifact_entity import (
        DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact,
        ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact,
        ClassificationMetricArtifact,
    )
    from src.entity.estimator import MyModel
    from src.entity.hf_estimator import HFModelEstimator
    assert MyModel is not None
    assert HFModelEstimator is not None


def test_cloud_and_config_imports():
    from src.configuration.mongo_db_connection import MongoDBClient
    from src.configuration.hf_connection import HFClient
    from src.cloud_storage.hf_storage import HuggingFaceStorage
    from src.data_access.LoanDefault_data import LoanDefaultData
    assert all([MongoDBClient, HFClient, HuggingFaceStorage, LoanDefaultData])


def test_pipeline_and_app_imports():
    from src.pipeline.training_pipeline import TrainPipeline
    from src.pipeline.prediction_pipeline import LoanApplicantData, LoanDefaultClassifier
    assert all([TrainPipeline, LoanApplicantData, LoanDefaultClassifier])