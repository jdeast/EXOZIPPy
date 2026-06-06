import click
import yaml
from .run import run_fit

@click.command()
@click.argument("config_file")
@click.option("--logger-level", default=None,
              type=click.Choice(["DEBUG", "INFO", "WARNING"], case_sensitive=False),
              help="Logging level (overrides logger_level in config file).")
def main(config_file, logger_level):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if logger_level:
        config["logger_level"] = logger_level.upper()

    run_fit(config)

if __name__ == "__main__":
    main()
