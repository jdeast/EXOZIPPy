#!/usr/bin/env python3
"""Thin CLI wrapper for exozippy.utilities.mkticsed.

The implementation moved to src/exozippy/utilities/mkticsed.py so components
and the GUI can discover and drive it generically. This wrapper keeps the
historical ``python scripts/mkticsed.py ...`` invocation working.
"""

from exozippy.utilities.mkticsed import main

if __name__ == "__main__":
    main()
