import os
import sys

from huggingface_hub import HfApi
from src.constants import HF_TOKEN_ENV_KEY
from src.exception import MyException
from src.logger import logging


class HFClient:
    """
    Sets up an authenticated HfApi client using the token from the environment.
    Mirrors the role S3Client/aws_connection.py would have played for AWS —
    a single place that handles authentication, reused by other components
    instead of each one re-authenticating on its own.
    """

    api = None
    def __init__(self):
        try:
            if HFClient.api is None:
                token = os.getenv(HF_TOKEN_ENV_KEY)
                if token is None:
                    raise Exception(
                        f"Environment variable '{HF_TOKEN_ENV_KEY}' is not set. "
                        f"Set it before running the pipeline."
                    )
                logging.info("Initializing Hugging Face Hub client")
                HFClient.api = HfApi(token=token)
                logging.info("Hugging Face Hub client initialized successfully")

            self.api = HFClient.api
        except Exception as e:
            raise MyException(e, sys)