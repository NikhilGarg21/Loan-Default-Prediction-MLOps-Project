"""
Confirms HFClient fails loudly and clearly when HF_TOKEN isn't set,
rather than silently proceeding with no authentication.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_hf_client_raises_when_token_missing(monkeypatch):
    # Ensure a clean slate regardless of the actual environment running this test
    monkeypatch.delenv("HF_TOKEN", raising=False)

    # Reset the class-level cache so this test doesn't reuse a token
    # set by an earlier import/instantiation in the same test session
    from src.configuration.hf_connection import HFClient
    HFClient.api = None

    with pytest.raises(Exception):
        HFClient()


def test_hf_client_succeeds_when_token_present(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_testing")

    from src.configuration.hf_connection import HFClient
    HFClient.api = None  # reset cache

    client = HFClient()
    assert client.api is not None