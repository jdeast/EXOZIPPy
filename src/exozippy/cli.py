import click

from .run import run_fit
from .yamlio import load_yaml


@click.command()
@click.argument("config_file")
@click.option(
    "--logger-level",
    default=None,
    type=click.Choice(["DEBUG", "INFO", "WARNING"], case_sensitive=False),
    help="Logging level (overrides logger_level in config file).",
)
def main(config_file, logger_level):
    # load_yaml, not yaml.safe_load: it refuses YAML-1.1-only boolean
    # spellings, which the GUI's ruamel loader would read as strings.
    config = load_yaml(config_file)

    if logger_level:
        config["logger_level"] = logger_level.upper()

    run_fit(config)


if __name__ == "__main__":
    main()
