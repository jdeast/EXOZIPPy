#!/usr/bin/env python3
"""Thin CLI wrapper for exozippy.utilities.mmexofast_to_params.

The implementation moved to src/exozippy/utilities/mmexofast_to_params.py so
components and the GUI can discover and drive it generically. This wrapper
keeps the historical ``python scripts/mmexofast_to_params.py ...`` invocation
working.
"""

from exozippy.utilities.mmexofast_to_params import main

if __name__ == "__main__":
    main()
