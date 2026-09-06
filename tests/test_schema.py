"""
Validates config/schema.yaml has the structure DataValidation and
DataTransformation actually depend on at runtime.
"""
import os
import yaml
import pytest

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "schema.yaml")


@pytest.fixture(scope="module")
def schema():
    with open(SCHEMA_PATH, "r") as f:
        return yaml.safe_load(f)


def test_schema_file_exists_and_parses():
    assert os.path.exists(SCHEMA_PATH), "config/schema.yaml is missing"


def test_required_top_level_keys(schema):
    required_keys = [
        "columns", "numerical_columns", "categorical_columns",
        "ordinal_columns", "binary_columns", "onehot_columns",
        "drop_columns", "sc_columns",
    ]
    for key in required_keys:
        assert key in schema, f"schema.yaml missing key: {key}"


def test_column_count_matches_dataset_shape(schema):
    """
    Loan_default.csv has 18 columns (16 features + LoanID + Default).
    If this drifts, DataValidation's column-count check will start failing
    for a reason that has nothing to do with the actual data.
    """
    assert len(schema["columns"]) == 18


def test_no_column_in_multiple_encoding_groups(schema):
    ordinal = set(schema.get("ordinal_columns", []))
    binary = set(schema.get("binary_columns", []))
    onehot = set(schema.get("onehot_columns", []))

    assert ordinal.isdisjoint(binary), f"Overlap (ordinal/binary): {ordinal & binary}"
    assert ordinal.isdisjoint(onehot), f"Overlap (ordinal/onehot): {ordinal & onehot}"
    assert binary.isdisjoint(onehot), f"Overlap (binary/onehot): {binary & onehot}"


def test_target_column_not_in_feature_groups(schema):
    """Default (the target) should never appear in the feature encoding groups."""
    for group in ["ordinal_columns", "binary_columns", "onehot_columns", "sc_columns"]:
        assert "Default" not in schema.get(group, []), f"Target column leaked into {group}"


def test_drop_columns_is_loan_id(schema):
    assert schema["drop_columns"] == "LoanID"