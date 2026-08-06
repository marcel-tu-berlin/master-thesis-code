"""FinQA reset must be seeded.

The shown question is read locally at seed % len while the server arms its
ground truth from the same seed. An unseeded reset armed a *random* server
question against the locally-shown row 0, so the episode was silently scored
against an answer the model never saw.
"""
import pytest

from domains.finqa.adapter import FinQAEnvAdapter


def test_reset_without_seed_raises():
    adapter = FinQAEnvAdapter("http://x", client=object())
    with pytest.raises(ValueError, match="seed"):
        adapter.reset()
