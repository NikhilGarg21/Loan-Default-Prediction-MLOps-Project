"""
Regression tests for MyModel's threshold-based prediction logic —
the exact bug found and fixed during this project (model silently
using its own default 0.5 cutoff instead of the locked threshold).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity.estimator import MyModel


class _FakeCatBoost:
    """Always returns a fixed P(class=1) = 0.4, so threshold behavior is fully predictable."""
    def predict_proba(self, X):
        n = len(X)
        return np.tile([0.6, 0.4], (n, 1))


class _FakePreprocessor:
    def transform(self, df):
        return df.values


def _sample_df():
    return pd.DataFrame({"a": [1, 2, 3]})


def test_default_threshold_is_point_five():
    model = MyModel(_FakePreprocessor(), _FakeCatBoost())
    assert model.threshold == 0.5


def test_threshold_above_probability_predicts_zero():
    model = MyModel(_FakePreprocessor(), _FakeCatBoost(), threshold=0.5)
    preds = model.predict(_sample_df())
    assert all(p == 0 for p in preds), "Probability 0.4 with threshold 0.5 should predict 0"


def test_threshold_below_probability_predicts_one():
    model = MyModel(_FakePreprocessor(), _FakeCatBoost(), threshold=0.3)
    preds = model.predict(_sample_df())
    assert all(p == 1 for p in preds), "Probability 0.4 with threshold 0.3 should predict 1"


def test_uses_predict_proba_not_raw_predict():
    """
    If someone 'simplifies' MyModel back to calling .predict() on the
    underlying model directly, this test catches it: _FakeCatBoost has
    no .predict() method at all, so calling it would raise AttributeError.
    """
    model = MyModel(_FakePreprocessor(), _FakeCatBoost(), threshold=0.4777)
    # Should not raise — proves predict_proba is what's actually being called
    preds = model.predict(_sample_df())
    assert preds is not None


def test_repr_includes_threshold():
    model = MyModel(_FakePreprocessor(), _FakeCatBoost(), threshold=0.4777)
    assert "0.4777" in repr(model)