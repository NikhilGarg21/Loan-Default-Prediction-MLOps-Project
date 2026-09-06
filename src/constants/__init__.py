import os
from datetime import date

DATABASE_NAME = "Loan_defaultDB"
COLLECTION_NAME = "Loan_defaultDB-Data"
MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "Default"
CURRENT_YEAR = date.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


HF_TOKEN_ENV_KEY = "HF_TOKEN"


# Data Ingestion related constant start with DATA_INGESTION VAR NAME

DATA_INGESTION_COLLECTION_NAME: str = "Loan_defaultDB-Data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.20


# Data Validation realted contant start with DATA_VALIDATION VAR NAME

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"


# Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


# MODEL TRAINER related constant start with MODEL_TRAINER var name

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_ITERATIONS: int = 700
MODEL_TRAINER_LEARNING_RATE: float = 0.03
MODEL_TRAINER_DEPTH: int = 4
MODEL_TRAINER_L2_LEAF_REG: int = 7
MODEL_TRAINER_SUBSAMPLE: float = 0.7
MODEL_TRAINER_RANDOM_STRENGTH: int = 2
MODEL_TRAINER_RANDOM_STATE: int = 42
MODEL_TRAINER_THRESHOLD: float = 0.4777

# MODEL Evaluation related constants

MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
HF_REPO_ID_ENV_KEY = "HF_REPO_ID"


AAPP_HOST = "0.0.0.0"
APP_PORT = int(os.getenv("PORT", 5000))