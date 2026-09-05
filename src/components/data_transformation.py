import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        StandardScaler on numeric columns + OneHotEncoder on the onehot columns,
        matching cells 20-25 of the notebook. Binary/ordinal columns are already
        mapped to numbers before this step, so they just pass through.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            numeric_transformer = StandardScaler()
            onehot_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

            sc_columns = self._schema_config['sc_columns']
            onehot_columns = self._schema_config['onehot_columns']
            logging.info("Cols loaded from schema.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, sc_columns),
                    ("OneHotEncoder", onehot_transformer, onehot_columns)
                ],
                remainder='passthrough'
            )

            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            return final_pipeline
        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e

    def _map_binary_columns(self, df):
        """Map No/Yes columns to 0/1 (HasMortgage, HasDependents, HasCoSigner)."""
        logging.info("Mapping binary columns to 0/1")
        for col in self._schema_config['binary_columns']:
            df[col] = df[col].map({"No": 0, "Yes": 1})
        return df

    def _map_ordinal_columns(self, df):
        """Map Education to an ordinal scale, matching the notebook."""
        logging.info("Mapping Education to ordinal scale")
        education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
        df["Education"] = df["Education"].map(education_map)
        return df

    def _drop_id_column(self, df):
        """Drop LoanID if it exists."""
        logging.info("Dropping id column")
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_final = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._map_binary_columns(input_feature_train_df)
            input_feature_train_df = self._map_ordinal_columns(input_feature_train_df)

            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._map_binary_columns(input_feature_test_df)
            input_feature_test_df = self._map_ordinal_columns(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_final = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            logging.info("Applying SMOTEEN for handling class imbalance in train data")
            smt = SMOTEENN(sampling_strategy='minority', random_state=42 , n_jobs=-1)
            input_feature_train_final, target_feature_train_final = smt.fit_resample(input_feature_train_arr, target_feature_train_df)
            logging.info("SMOTEEN applied successfully")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise MyException(e, sys) from e
