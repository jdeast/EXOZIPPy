import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--plot-grids",
        action="store_true",
        default=False,
        help="Display plots of test grids with minima marked"
    )
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Skip slow tests that call searcher.run() on real data.",
    )

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: calls searcher.run() on real data; omit with --fast",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--fast"):
        skip_slow = pytest.mark.skip(reason="skipped by --fast flag")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
