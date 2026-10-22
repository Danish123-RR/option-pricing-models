from src.black_scholes import bs_price
from src.monte_carlo import mc_gbm


def test_mc_brackets_black_scholes():
    exact = bs_price(100, 100, 1, 0.05, 0.2)
    res = mc_gbm(100, 100, 1, 0.05, 0.2, n_paths=400_000, seed=1)
    low, high = res.ci95
    assert low < exact < high


def test_antithetic_reduces_error():
    plain = mc_gbm(100, 100, 1, 0.05, 0.2, n_paths=200_000, antithetic=False, seed=2)
    anti = mc_gbm(100, 100, 1, 0.05, 0.2, n_paths=200_000, antithetic=True, seed=2)
    assert anti.std_error < plain.std_error


def test_error_shrinks_like_one_over_root_n():
    small = mc_gbm(100, 100, 1, 0.05, 0.2, n_paths=10_000, antithetic=False, seed=3)
    large = mc_gbm(100, 100, 1, 0.05, 0.2, n_paths=160_000, antithetic=False, seed=3)
    ratio = small.std_error / large.std_error   # 16x the paths -> about 4x smaller error
    assert 3.5 < ratio < 4.5
