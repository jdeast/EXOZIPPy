#!/usr/bin/env python3
"""Thin CLI wrapper for exozippy.utilities.exofast2exozippy.

The implementation moved to src/exozippy/utilities/exofast2exozippy.py so it
installs as the ``exozippy-exofast2exozippy`` command. This wrapper keeps the
historical ``python scripts/exofast2exozippy.py ...`` invocation working.
"""

from exozippy.utilities.exofast2exozippy import main

if __name__ == "__main__":
    main()
