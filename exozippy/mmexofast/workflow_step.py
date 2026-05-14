# workflow_step.py
from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    PENDING  = auto()
    SKIPPED  = auto()
    RUNNING  = auto()
    COMPLETE = auto()
    FAILED   = auto()


class WorkflowStep:
    """
    A single, named unit of work within a larger fitting workflow.

    Parameters
    ----------
    name : str
        Unique identifier used for dependency resolution and logging.
    func : Callable
        The callable that performs the actual work.  It will be invoked
        as ``func()`` (no arguments) by :meth:`run`.  All parameters
        the callable needs must be captured at construction time via
        closure or ``functools.partial``.
    stage : str, optional
        Logical grouping label for this step.  Multiple steps that
        belong to the same phase of the workflow (e.g. all parallax
        branch fits) share the same stage name.  Used by tests and
        progress reporters to reason about workflow position without
        relying on individual step names.
    description : str, optional
        Human-readable summary shown in progress output.
    dependencies : list of str, optional
        Names of steps that must reach COMPLETE status before this step
        is allowed to run.
    required : bool
        If True (default) a FAILED status aborts the workflow.
        If False the workflow continues and this step is logged as a
        warning rather than an error.
    max_retries : int
        How many additional attempts to make after the first failure
        before marking the step as permanently FAILED.
    """

    def __init__(
        self,
        name: str,
        func: Callable[[], Any],
        *,
        stage: str = "",
        description: str = "",
        dependencies: Optional[list[str]] = None,
        required: bool = True,
        max_retries: int = 0,
    ) -> None:
        self.name         = name
        self.func         = func
        self.stage        = stage
        self.description  = description
        self.dependencies = dependencies or []
        self.required     = required
        self.max_retries  = max_retries

        self.status:   StepStatus          = StepStatus.PENDING
        self.result:   Any                 = None
        self.error:    Optional[Exception] = None
        self._attempts: int                = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> Any:
        """
        Execute the step, honoring the retry policy.

        Calls ``self.func()``, stores the return value in
        ``self.result``, and sets ``self.status`` to ``COMPLETE``.
        On failure, retries up to ``self.max_retries`` times before
        setting ``self.status`` to ``FAILED`` and re-raising the
        exception.

        Returns
        -------
        Any
            The return value of ``self.func``.

        Raises
        ------
        Exception
            Re-raises the last exception if all attempts fail and
            ``self.required`` is True.
        """
        self.status = StepStatus.RUNNING

        for attempt in range(1 + self.max_retries):
            self._attempts += 1
            try:
                self.result = self.func()
                self.status = StepStatus.COMPLETE
                logger.debug("Step '%s' complete.", self.name)
                return self.result
            except Exception as exc:
                self.error = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "Step '%s' failed on attempt %d/%d: %s.  "
                        "Retrying.",
                        self.name,
                        attempt + 1,
                        1 + self.max_retries,
                        exc,
                    )
                else:
                    self.status = StepStatus.FAILED
                    logger.error(
                        "Step '%s' failed after %d attempt(s): %s.",
                        self.name,
                        self._attempts,
                        exc,
                    )
                    if self.required:
                        raise

        return self.result

    def skip(self, reason: str = "") -> None:
        """
        Mark the step SKIPPED without executing it.

        Parameters
        ----------
        reason : str, optional
            Human-readable explanation logged at DEBUG level.
        """
        self.status = StepStatus.SKIPPED
        if reason:
            logger.debug("Step '%s' skipped: %s", self.name, reason)
        else:
            logger.debug("Step '%s' skipped.", self.name)

    def reset(self) -> None:
        """
        Return the step to its initial PENDING state.

        Clears ``result``, ``error``, and ``_attempts``.
        """
        self.status    = StepStatus.PENDING
        self.result    = None
        self.error     = None
        self._attempts = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True if the step has completed successfully."""
        return self.status is StepStatus.COMPLETE

    @property
    def has_failed(self) -> bool:
        """True if the step has failed all attempts."""
        return self.status is StepStatus.FAILED

    @property
    def is_pending(self) -> bool:
        """True if the step has not yet been attempted."""
        return self.status is StepStatus.PENDING

    @property
    def attempts(self) -> int:
        """Number of execution attempts made so far."""
        return self._attempts

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"WorkflowStep(name={self.name!r}, "
            f"stage={self.stage!r}, "
            f"status={self.status.name}, "
            f"required={self.required})"
        )
