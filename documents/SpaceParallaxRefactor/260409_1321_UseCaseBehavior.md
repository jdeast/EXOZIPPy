Here are four use cases covering the main workflows:

---

## Use Case 1: Ground-Only Point-Lens Fit (One-Touch)

**User action:**
```python
fitter = MMEXOFASTFitter(
    files=['OGLE.txt'],
    coords='17:47:12.25 -21:22:58.7',
    fit_type='point lens',
    finite_source=False,
    renormalize_errors=True,
    parallax_grid=True,
    output_config=OutputConfig(
        base_dir='output', file_head='ob0939',
        save_log=True, save_plots=True,
        save_latex_tables=True, save_restart_files=True,
        save_grid_results=True)
)
fitter.fit()
```

**Expected behavior:**

1. `__init__` completes with `workflow_stage="start"`
2. `fit()` is called
3. `_mark_stale_fits()` runs — no fits exist, nothing to mark
4. No new locations detected
5. Workflow executes in sequence:
   - `"start"` → `"event_search_done"`: EF grid search runs on OGLE data, `best_ef_grid_point` stored, restart saved
   - `"event_search_done"` → `"static_fit_done"`: PSPL fit runs seeded from EF grid, restart saved
   - `"static_fit_done"` → `"pl_fit_done"`: PAR_u0+ and PAR_u0− fits run seeded from PSPL, restart saved
   - `"pl_fit_done"` → `"primary_renorm_done"`: best model selected, OGLE renormalized, all fits marked `needs_refit=True` and refit, parallax grid runs, restart saved
   - No alternate locations → `"complete"`
6. `_check_renorm_completeness()` runs — all datasets renormalized, no warning
7. LaTeX table written with all fits
8. Log written throughout

**Expected outputs:**
- `ob0939.log`
- `ob0939_ef_grid.png`
- `ob0939_piE_grid.png`
- `ob0939_piE_grid_u0_plus.txt`, `ob0939_piE_grid_u0_minus.txt`
- `ob0939_results.tex` with PSPL, PAR_u0+, PAR_u0− columns
- `ob0939_restart.pkl` with `workflow_stage="complete"`

---

## Use Case 2: Incremental Ground-Only Workflow (Three Steps)

**Step 1 — Raw fit:**
```python
fitter = MMEXOFASTFitter(
    files=['OGLE.txt'],
    coords='17:47:12.25 -21:22:58.7',
    fit_type='point lens',
    finite_source=False,
    renormalize_errors=False,
    output_config=OutputConfig(
        base_dir='output', file_head='ob0939_raw',
        save_log=True, save_latex_tables=True,
        save_restart_files=True)
)
fitter.fit()
```

Expected: Runs EF grid, fits PSPL, PAR_u0+, PAR_u0−. No renormalization. Stage reaches `"complete"`. Restart saved.

---

**Step 2 — Restart and renormalize:**
```python
fitter = MMEXOFASTFitter(
    restart_file='output/ob0939_raw_restart.pkl',
    finite_source=False,
    renormalize_errors=True,
    parallax_grid=True,
    output_config=OutputConfig(
        base_dir='output', file_head='ob0939_renorm',
        save_log=True, save_plots=True,
        save_latex_tables=True, save_restart_files=True,
        save_grid_results=True)
)
fitter.fit()
```

Expected behavior:
1. `__init__` loads restart file — `workflow_stage="complete"`, existing fits loaded
2. `fit()` called
3. `_mark_stale_fits()` runs — renorm factors unchanged, no new datasets, no fits marked
4. R2.15 settings-change check runs — `renormalize_errors` changed from `False` to `True` → stage reset to `"pl_fit_done"`, all fits marked `needs_refit=True`
5. No new locations detected
6. Workflow resumes from `"pl_fit_done"`:
   - `"pl_fit_done"` → `"primary_renorm_done"`: OGLE renormalized using best current model, all fits refit, parallax grid runs
   - No alternate locations → `"complete"`
7. LaTeX table written, log written

---

**Step 3 — Add followup data:**
```python
fitter = MMEXOFASTFitter(
    files=['OGLE.txt', 'MOA.txt'],
    restart_file='output/ob0939_renorm_restart.pkl',
    finite_source=False,
    renormalize_errors=True,
    parallax_grid=True,
    output_config=OutputConfig(
        base_dir='output', file_head='ob0939_full_ground',
        save_log=True, save_plots=True,
        save_latex_tables=True, save_restart_files=True,
        save_grid_results=True)
)
fitter.fit()
```

Expected behavior:
1. `__init__` loads restart file — OGLE dataset replaced with saved (renormalized) version, MOA loaded fresh
2. `workflow_stage="complete"` from restart file
3. `fit()` called
4. `_mark_stale_fits()` runs — MOA is new same-location dataset, all ground fits marked `needs_refit=True`, stage reset to `"event_search_done"`
5. R2.15 check runs — renormalize_errors changed from False to True → stage reset to "pl_fit_done", all fits marked needs_refit=True
6. Two-pass same-location workflow executes:
   - Pass 1: EF grid skipped (still valid), static fit through parallax fits rerun with OGLE (renorm'd) + MOA (unrenorm'd)
   - Pass 2: MOA renormalized using best current model, all fits marked `needs_refit=True` and refit, parallax grid runs
7. No alternate locations → `"complete"`
8. LaTeX table written — all fits included

---

## Use Case 3: Incremental Ground + Space Workflow

**Step 1 — Ground fit with renormalization:**
*(Same as Use Case 2, Step 1 + Step 2 combined via one-touch — produces `ob0939_ground_restart.pkl` at `"complete"`)*

---

**Step 2 — Add Spitzer:**
```python
fitter = MMEXOFASTFitter(
    files=['OGLE.txt', 'Spitzer.txt'],
    restart_file='output/ob0939_ground_restart.pkl',
    finite_source=False,
    renormalize_errors=True,
    parallax_grid=True,
    output_config=OutputConfig(
        base_dir='output', file_head='ob0939_complete',
        save_log=True, save_plots=True,
        save_latex_tables=True, save_restart_files=True,
        save_grid_results=True)
)
fitter.fit()
```

Expected behavior:
1. `__init__` loads restart — OGLE replaced with saved (renormalized) version, Spitzer loaded fresh
2. `workflow_stage="complete"` from restart
3. `fit()` called
4. `_mark_stale_fits()` runs — Spitzer is a new location; existing ground fits NOT marked stale
5. R2.15 settings-change check runs — no settings changed, no-op
6. New location detected — stage reset to `"primary_renorm_done"` per R3.1.8; single-pass different-location workflow executes:
   - Static PSPL fit with all data (OGLE + Spitzer)
   - Coarse parallax grid runs seeded from _find_related_fit() — no n_loc=2 fits exist yet, so falls back to _select_preferred_point_lens(), returning best ground parallax fit (PAR_u0+ or PAR_u0−); minima optimized; duplicates merged; solution_index assigned
   - Spitzer renormalized using best multi-location model
   - Multi-location fits marked `needs_refit=True` and refit
   - Final parallax grid runs (since `parallax_grid=True`)
7. Stage reaches `"complete"`
8. Log notes: ground-only fits preserved, multi-location fits added
9. LaTeX table includes both ground-only fits (PSPL, PAR_u0+, PAR_u0−) and multi-location fits (by solution index)

**Key check:** Ground-only PAR_u0+, PAR_u0− fits from Step 1 must still appear in the table and must NOT have `needs_refit=True`.

---

## Use Case 4: Ground-Only Binary Lens Fit (One-Touch)

```python
fitter = MMEXOFASTFitter(
    files=['OGLE.txt', 'MOA.txt'],
    coords='17:47:12.25 -21:22:58.7',
    fit_type='binary lens',
    finite_source=False,
    renormalize_errors=True,
    parallax_grid=False,
    output_config=OutputConfig(
        base_dir='output', file_head='ob0939_binary',
        save_log=True, save_latex_tables=True,
        save_restart_files=True)
)
fitter.fit()
```

**Expected behavior:**

1. `__init__` completes with `workflow_stage="start"`
2. `fit()` called, routes to `fit_binary_lens()`
3. `fit_binary_lens()` calls `fit_point_lens(stop_at="primary_renorm_done")`:
   - EF grid → static PSPL → PAR_u0+, PAR_u0− → renormalization → stage reaches `"primary_renorm_done"`
4. Binary stages execute:
   - `"primary_renorm_done"` → `"anomaly_search_done"`: residuals computed from best PL model, AnomalyFinder runs on ground data only, `best_af_grid_point` stored
   - `"anomaly_search_done"` → `"binary_fit_done"`: binary fit runs seeded from `anomaly_lc_params`, duplicate solutions merged, `solution_index` assigned, log reports original vs. merged count
   - `"binary_fit_done"` → `"post_binary_renorm_done"`: `_needs_renormalization()` called — returns `False` (threshold disabled), no re-renormalization
   - No alternate locations → `"complete"`
5. `_check_renorm_completeness()` — all datasets renormalized, no warning
6. LaTeX table includes PSPL, PAR_u0+, PAR_u0−, and binary solution(s)

**Key check:** If log shows e.g. "13 minima found, merged to 4 solutions", the table must show exactly 4 binary fits labeled by `solution_index` (0, 1, 2, 3), with index 0 having the lowest chi2.

---

These four use cases cover: one-touch, incremental restart, same-location data addition, different-location data addition, and binary lens. Each maps directly to requirements in the spec so you can trace expected behavior to specific R-numbers during your developer review.