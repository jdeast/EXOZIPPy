import click
import yaml
from .run import run_fit

@click.command()
@click.argument("config_file")
def main(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Call the library function
    run_fit(config)