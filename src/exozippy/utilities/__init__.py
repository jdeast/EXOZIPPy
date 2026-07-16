"""Importable component utilities and a component-agnostic registry.

Each utility (getdata, mkticsed, mmexofast_to_params, ...) lives in its own
module exposing ``build_parser() -> argparse.ArgumentParser`` and
``main(argv=None)``. Components declare which utilities they surface via
``Component.get_utilities()``; the registry converts an argparse parser into a
JSON-serializable argument schema and runs utilities headlessly. See
:mod:`exozippy.utilities.registry`.
"""

from .registry import (
    UtilitySpec,
    all_utilities,
    argparse_subprocess_runner,
    args_dict_to_argv,
    inprocess_runner,
    parser_to_schema,
    run_utility,
)

__all__ = [
    "UtilitySpec",
    "all_utilities",
    "argparse_subprocess_runner",
    "args_dict_to_argv",
    "inprocess_runner",
    "parser_to_schema",
    "run_utility",
]
