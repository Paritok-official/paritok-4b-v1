"""Prompt-cache read multiplier for the MiniMax text models.

MiniMax bills an input token served from its prompt cache at 20% of the base
input price (M3: $0.12/M read against $0.60/M input; M2.7: $0.06/M against
$0.30/M). Without an entry in CACHE_READ_MULT these ids fall through to
DEFAULT_CACHE_READ_MULT (0.1x), which halves the cached-input saving that
/stats reports for every turn after the first.
"""
import pytest

from paritok.proxy.pricing import DEFAULT_CACHE_READ_MULT, cache_read_multiplier


@pytest.mark.parametrize(
    "model",
    [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "minimax-m3",
        "minimax-m2.7",
        "minimax/MiniMax-M3",  # provider-namespaced id
        "minimax/MiniMax-M2.7",
    ],
)
def test_minimax_cache_read_is_20_percent_of_base_input(model):
    assert cache_read_multiplier(model) == pytest.approx(0.2)


def test_minimax_does_not_fall_back_to_the_unknown_model_default():
    assert cache_read_multiplier("MiniMax-M3") != pytest.approx(DEFAULT_CACHE_READ_MULT)


def test_unknown_model_still_uses_the_default_multiplier():
    assert cache_read_multiplier("some-unlisted-model") == pytest.approx(
        DEFAULT_CACHE_READ_MULT
    )
