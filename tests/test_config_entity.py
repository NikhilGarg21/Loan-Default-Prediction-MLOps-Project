"""
Sanity checks on config_entity.py's dataclasses — catches accidental
constant mixups (like the HF_REPO_ID-as-a-literal-string bug found
earlier) and out-of-range hyperparameter values.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity.config_entity import (
    ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig,
    DataIngestionConfig,
)


def test_model_trainer_hyperparameters_in_sane_ranges():
    cfg = ModelTrainerConfig()
    assert 0 < cfg._learning_rate < 1
    assert cfg._iterations > 0
    assert cfg._depth > 0
    assert 0 < cfg._subsample <= 1
    assert 0 <= cfg._model_threshold <= 1, "Threshold must be a valid probability cutoff"


def test_model_trainer_threshold_is_not_default_point_five():
    """
    This project deliberately calibrated a non-default threshold (0.4777).
    If this ever silently reverts to 0.5, it's worth knowing — that would
    mean recalibration got lost, not necessarily that 0.5 is wrong.
    """
    cfg = ModelTrainerConfig()
    assert cfg._model_threshold != 0.5, (
        "Threshold is back to the untuned default — was recalibration lost?"
    )


def test_hf_repo_id_is_not_a_literal_placeholder():
    """
    Regression test for the exact bug found earlier: HF_REPO_ID was
    accidentally set to the literal string 'HF_REPO_ID' instead of
    being read from the environment via os.getenv().
    """
    cfg = ModelEvaluationConfig()
    assert cfg.hf_repo_id != "HF_REPO_ID", (
        "hf_repo_id is literally the string 'HF_REPO_ID' — "
        "check config_entity.py uses os.getenv(HF_REPO_ID_ENV_KEY), not the raw constant name"
    )

    pusher_cfg = ModelPusherConfig()
    assert pusher_cfg.hf_repo_id != "HF_REPO_ID"


def test_train_test_split_ratio_is_reasonable():
    cfg = DataIngestionConfig()
    assert 0 < cfg.train_test_split_ratio < 1


def test_model_evaluation_threshold_score_is_positive():
    cfg = ModelEvaluationConfig()
    assert cfg.changed_threshold_score > 0, (
        "A zero or negative threshold would accept any new model, even a worse one"
    )