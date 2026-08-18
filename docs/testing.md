# Testing conventions

Tests follow AAA (Arrange / Act / Assert) with Given/When/Then docstrings. All tests that use `System` must call `system.prepare()` before `system.build_model()`. RA/Dec user params are in **degrees** (the default unit); `Parameter.__post_init__` converts to radians internally.

Do not start the full suite with a timeout. Start it and poll.

Testing note: build relation inputs with `pt.dscalar`, **not** `pt.as_tensor_variable(<python float>)` -- pytensor autocasts a bare Python float to the smallest dtype that represents it (5778.0 -> float32), and a unary op like `pt.log10` on it then computes in float32, silently losing ~1e-7. The model always feeds float64. `tests/test_torres.py` pins the port against real IDL output from `massradius_torres.pro`.

## Suite runtime and the pytensor compile cache (PLACEHOLDER -- owned by another branch)

The runbook for the pytensor compile cache, and the suite's real wall-clock cost, are being
rewritten by the work that fixes the underlying cache bug; that text will be merged in here
as `docs/testing-cache.md`. Do not invent a policy in the meantime.

Two claims that used to live in `CLAUDE.md` are wrong and are recorded here only so nobody
reintroduces them:

- "The test suite takes ~10 minutes." Measured ~9:39 warm, but the compile-cache pathology
  makes it much worse in practice, so the number is not a useful expectation.
- "pytest Timeout inside pytensor cmodule.py, blaming an innocent test: the compile cache
  grew until refresh(), which opens every entry, neared the 300s cap. Its tmp*/ dirs ARE
  the cached modules -- prune by age, never rm them. `poetry run pytensor-cache cleanup`."
  The diagnosis of the Timeout is right; the remedy is not. `pytensor-cache cleanup` only
  deletes entries older than 31 days and provably does nothing here (measured 4035 -> 4034
  entries, 4.1 G -> 4.1 G).
