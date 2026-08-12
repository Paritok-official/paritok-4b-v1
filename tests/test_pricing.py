import pytest

from paritok.proxy.pricing import input_usd_per_mtok


@pytest.mark.parametrize(
    ("model", "expected_price"),
    [
        ("MiniMax-M3", 0.60),
        ("MiniMax-M2.7", 0.30),
        ("minimax/MiniMax-M3", 0.60),
        ("minimax/MiniMax-M2.7", 0.30),
    ],
)
def test_minimax_input_prices(model, expected_price):
    assert input_usd_per_mtok(model) == (expected_price, True)
