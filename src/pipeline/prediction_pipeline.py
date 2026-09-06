import sys
from src.entity.config_entity import LoanDefaultPredictorConfig
from src.entity.hf_estimator import HFModelEstimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame
import os

class LoanApplicantData:
    def __init__(self,
                 Age,
                 Income,
                 LoanAmount,
                 CreditScore,
                 MonthsEmployed,
                 NumCreditLines,
                 InterestRate,
                 LoanTerm,
                 DTIRatio,
                 Education,
                 EmploymentType,
                 MaritalStatus,
                 HasMortgage,
                 HasDependents,
                 LoanPurpose,
                 HasCoSigner
                 ):
        """
        LoanApplicantData constructor
        Input: all features of the trained model for prediction

        Note: Education, HasMortgage, HasDependents, and HasCoSigner are expected
        already encoded (ordinal / 0-1) here, matching what DataTransformation
        produces before the ColumnTransformer runs. EmploymentType, MaritalStatus,
        and LoanPurpose are expected as their original string categories, since
        OneHotEncoder inside the saved pipeline expects those raw string values.
        """
        try:
            self.Age = Age
            self.Income = Income
            self.LoanAmount = LoanAmount
            self.CreditScore = CreditScore
            self.MonthsEmployed = MonthsEmployed
            self.NumCreditLines = NumCreditLines
            self.InterestRate = InterestRate
            self.LoanTerm = LoanTerm
            self.DTIRatio = DTIRatio
            self.Education = Education
            self.EmploymentType = EmploymentType
            self.MaritalStatus = MaritalStatus
            self.HasMortgage = HasMortgage
            self.HasDependents = HasDependents
            self.LoanPurpose = LoanPurpose
            self.HasCoSigner = HasCoSigner
        except Exception as e:
            raise MyException(e, sys) from e

    def get_loan_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame from LoanApplicantData class input
        """
        try:
            loan_input_dict = self.get_loan_data_as_dict()
            return DataFrame(loan_input_dict)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_loan_data_as_dict(self):
        """
        This function returns a dictionary from LoanApplicantData class input
        """
        logging.info("Entered get_loan_data_as_dict method of LoanApplicantData class")
        try:
            input_data = {
                "Age": [self.Age],
                "Income": [self.Income],
                "LoanAmount": [self.LoanAmount],
                "CreditScore": [self.CreditScore],
                "MonthsEmployed": [self.MonthsEmployed],
                "NumCreditLines": [self.NumCreditLines],
                "InterestRate": [self.InterestRate],
                "LoanTerm": [self.LoanTerm],
                "DTIRatio": [self.DTIRatio],
                "Education": [self.Education],
                "EmploymentType": [self.EmploymentType],
                "MaritalStatus": [self.MaritalStatus],
                "HasMortgage": [self.HasMortgage],
                "HasDependents": [self.HasDependents],
                "LoanPurpose": [self.LoanPurpose],
                "HasCoSigner": [self.HasCoSigner]
            }
            logging.info("Created loan applicant data dict")
            logging.info("Exited get_loan_data_as_dict method of LoanApplicantData class")
            return input_data
        except Exception as e:
            raise MyException(e, sys) from e


class LoanDefaultClassifier:
    def __init__(self, prediction_pipeline_config: LoanDefaultPredictorConfig = LoanDefaultPredictorConfig()) -> None:
        """
        :param prediction_pipeline_config: Configuration for predicting the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of LoanDefaultClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of LoanDefaultClassifier class")
            model = HFModelEstimator(
                repo_id=self.prediction_pipeline_config.hf_repo_id,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(dataframe)
            return result
        except Exception as e:
            raise MyException(e, sys)