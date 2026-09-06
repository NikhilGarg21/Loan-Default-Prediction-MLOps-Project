import os
import sys
import pickle

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from pandas import DataFrame, read_csv

from src.configuration.hf_connection import HFClient
from src.logger import logging
from src.exception import MyException

class HuggingFaceStorage:
    """
    A class for interacting with Hugging Face Hub, providing methods for file
    management, model uploads/retrieval, and general dataset storage.
    Mirrors the role SimpleStorageService played for AWS S3.

    Model-related methods use repo_type="model"; general file/CSV methods
    default to repo_type="dataset", HF's equivalent of a general-purpose bucket.
    """

    def __init__(self):
        hf_client = HFClient()
        self.api = hf_client.api

    def model_path_available(self, repo_id: str, path_in_repo: str) -> bool:
        """Checks if a specified model file is available in the given HF model repo."""
        try:
            files = self.api.list_repo_files(repo_id=repo_id, repo_type="model")
            return path_in_repo in files
        except RepositoryNotFoundError:
            return False
        except Exception as e:
            raise MyException(e, sys)

    def load_model(self, model_name: str, repo_id: str, model_dir: str = None) -> object:
        """Loads a serialized model from the specified Hugging Face model repo."""
        try:
            path_in_repo = f"{model_dir}/{model_name}" if model_dir else model_name
            local_path = hf_hub_download(repo_id=repo_id, filename=path_in_repo, repo_type="model")
            with open(local_path, "rb") as f:
                model = pickle.load(f)
            logging.info("Production model loaded from Hugging Face Hub.")
            return model
        except Exception as e:
            raise MyException(e, sys)

    def create_repo(self, repo_id: str, repo_type: str = "model") -> None:
        """Creates a Hugging Face repo (model or dataset) if it doesn't already exist."""
        logging.info("Entered the create_repo method of HuggingFaceStorage class")
        try:
            self.api.create_repo(repo_id=repo_id, exist_ok=True, repo_type=repo_type)
            logging.info("Exited the create_repo method of HuggingFaceStorage class")
        except Exception as e:
            raise MyException(e, sys)

    def upload_file(self, from_filename: str, to_filename: str, repo_id: str,
                     repo_type: str = "model", remove: bool = True):
        """Uploads a local file to the specified Hugging Face repo."""
        logging.info("Entered the upload_file method of HuggingFaceStorage class")
        try:
            logging.info(f"Uploading {from_filename} to {to_filename} in {repo_id}")
            self.create_repo(repo_id, repo_type=repo_type)
            self.api.upload_file(
                path_or_fileobj=from_filename,
                path_in_repo=to_filename,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            logging.info(f"Uploaded {from_filename} to {to_filename} in {repo_id}")

            if remove:
                os.remove(from_filename)
                logging.info(f"Removed local file {from_filename} after upload")
            logging.info("Exited the upload_file method of HuggingFaceStorage class")
        except Exception as e:
            raise MyException(e, sys)

    # ---------- General-purpose: can be used for any repo type ----------
    def file_path_available(self, repo_id: str, path_in_repo: str, repo_type: str = "dataset") -> bool:
        """General-purpose version of model_path_available, for any repo type."""
        try:
            files = self.api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
            return path_in_repo in files
        except RepositoryNotFoundError:
            return False
        except Exception as e:
            raise MyException(e, sys)

    def download_file(self, filename: str, repo_id: str, repo_type: str = "dataset") -> str:
        """Downloads any file (not just models) from a repo, returns the local cached path."""
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
            return local_path
        except Exception as e:
            raise MyException(e, sys)

    def upload_df_as_csv(self, data_frame: DataFrame, local_filename: str,
                          repo_filename: str, repo_id: str, repo_type: str = "dataset") -> None:
        """Uploads a DataFrame as a CSV file to the specified Hugging Face repo."""
        logging.info("Entered the upload_df_as_csv method of HuggingFaceStorage class")
        try:
            data_frame.to_csv(local_filename, index=None, header=True)
            self.upload_file(local_filename, repo_filename, repo_id, repo_type=repo_type)
            logging.info("Exited the upload_df_as_csv method of HuggingFaceStorage class")
        except Exception as e:
            raise MyException(e, sys)

    def read_csv(self, filename: str, repo_id: str, repo_type: str = "dataset") -> DataFrame:
        """Reads a CSV file from a Hugging Face repo and returns it as a DataFrame."""
        logging.info("Entered the read_csv method of HuggingFaceStorage class")
        try:
            local_path = self.download_file(filename, repo_id, repo_type=repo_type)
            df = read_csv(local_path, na_values="na")
            logging.info("Exited the read_csv method of HuggingFaceStorage class")
            return df
        except Exception as e:
            raise MyException(e, sys)