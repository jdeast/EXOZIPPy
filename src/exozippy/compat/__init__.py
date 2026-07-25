"""Narrow, self-retiring workarounds for upstream bugs.

Everything here patches a third-party package to get around a defect we do not
own.  Each patch must:

  * detect the defect rather than the version, so it disables itself the
    moment the upstream fix lands (a version pin would silently keep
    shadowing upstream long after it was fixed);
  * be idempotent, so repeated application is harmless;
  * name the upstream defect and the condition for deleting it.

Nothing in ``exozippy`` proper should import a private third-party symbol; if
you need one, wrap it here.
"""

from .blackjax_progressbar import patch_blackjax_progress_bar

__all__ = ["patch_blackjax_progress_bar"]
