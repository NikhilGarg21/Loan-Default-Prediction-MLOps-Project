import sys

from src.cloud_storage.hf_storage import HuggingFaceStorage
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.hf_estimator import HFModelEstimator

class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.hf = HuggingFaceStorage()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.hf_estimator = HFModelEstimator(repo_id=model_pusher_config.hf_repo_id,
                                              model_path=model_pusher_config.hf_model_path)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This function is used to initiate all steps of the model pusher

        Output      :   Returns model pusher artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            logging.info("Uploading model to Hugging Face Hub repo")
            self.hf_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            model_pusher_artifact = ModelPusherArtifact(hf_repo_id=self.model_pusher_config.hf_repo_id,
                                                         hf_model_path=self.model_pusher_config.hf_model_path)
            logging.info("Uploaded model to Hugging Face Hub repo")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e