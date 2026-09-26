THE ASYNC `tight` ARM'S OUTPUTS WERE OVERWRITTEN ON 2026-09-16 AND ARE GONE.

Cause: ab194/tight_sync was cloned from this directory by copying the config
and changing ONLY `parameter_file`.  The config's `prefix:` key -- which is
what every output file is named from -- still pointed here, so the
tight_sync run wrote its log, trace, results.csv, summary.txt, modes.txt
and every plot on top of this arm's.  The SIGTERM that was sent to this arm
specifically to PRESERVE its trace for forensics was therefore wasted.

Lost: DC2018_194_trace.nc (2.71 GB, 30,896 draws/chain, the graceful-stop
trace), plus this arm's results.csv, summary.txt, modes.txt and log.

Surviving record of what it said, quoted in notes/supercomputer_queue.txt:
  score        core 6/7, all 11/15, mixing OK (truth table tier 'default')
  chains kept  78, WITH the "<3 chains reached the good-likelihood region;
               all chains kept (possible stuck-chain contamination)" note
  diag_convergence  SLOW/ok, lp drift 0.9, mode sep 0.4 (source.u_0)
  pulls        D_source -5.09, R_source -13.64, t_0 +2.62, t_E -0.11
What cannot be recovered: the per-chain lp history, i.e. the forensics the
trace was kept for.

ab194/u0te's equivalents were copied to ab194/_async_preserved/u0te/ before
the same thing could happen there.  Its .log was already being rewritten by
then, so the copy of THAT file is the sync run's, not the async one's.

Both *_sync configs now carry their own `prefix:` and sed path.  When
cloning an arm, re-point EVERY absolute path in the config, not just the
one that looks like the input.
