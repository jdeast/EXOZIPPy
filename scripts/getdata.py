#!/usr/bin/env python3
"""Thin CLI wrapper for exozippy.utilities.getdata.

The implementation moved to src/exozippy/utilities/getdata.py so components
and the GUI can discover and drive it generically. This wrapper keeps the
historical ``python scripts/getdata.py ...`` invocation working.
"""

from exozippy.utilities.getdata import main

if __name__ == "__main__":
    main()
