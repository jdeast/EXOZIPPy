def pytest_addoption(parser):
    parser.addoption(
        "--plot-grids",
        action="store_true",
        default=False,
        help="Display plots of test grids with minima marked"
    )
