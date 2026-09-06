import sys

from pandas import DataFrame

from src.cloud_storage.hf_storage import HuggingFaceStorage
from src.exception import MyException
from src.entity.estimator import MyModel

class HFModelEstimator:
    """
    This class is used to save and retrieve our model from Hugging Face Hub
    and to do prediction.
    """

    def __init__(self, repo_id, model_path):
        """
        :param repo_id: Hugging Face repo id, e.g. "username/loan-default-model"
        :param model_path: Filename of the model within the repo
        """
        self.repo_id = repo_id
        self.hf = HuggingFaceStorage()
        self.model_path = model_path
        self.loaded_model: MyModel = None

    def is_model_present(self, model_path):
        try:
            return self.hf.model_path_available(repo_id=self.repo_id, path_in_repo=model_path)
        except MyException as e:
            print(e)
            return False

    def load_model(self) -> MyModel:
        """
        Load the model from the model_path
        :return:
        """
        return self.hf.load_model(self.model_path, repo_id=self.repo_id)

    def save_model(self, from_file, remove: bool = False) -> None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            self.hf.upload_file(
                from_file,
                to_filename=self.model_path,
                repo_id=self.repo_id,
                remove=remove
            )
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: DataFrame):
        """
        :param dataframe:
        :return:
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise MyException(e, sys)