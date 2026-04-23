## Section 1: Data Structures — Specification

---

### 1.1 `FitKey`

#### Background
`FitKey` is a frozen dataclass used as a dictionary key in `AllFitResults`. It identifies a fit by its model type. Currently it encodes parallax branch as an enum (U0_PLUS, U0_MINUS, U0_PP, U0_PM, U0_MP, U0_MM). The PP/PM/MP/MM branches encode the u0 sign combination for ground and space locations — but this assignment requires knowing the space trajectory, which can fail if the optimizer wanders outside the satellite ephemeris window. The new design defers branch label assignment to output time; during fitting, multi-location solutions are identified by a sequential integer index instead.

#### Requirements

**R1.1.1** `FitKey` must have a `solution_index` field of type `Optional[int]`, defaulting to `None`.

**R1.1.2** `parallax_branch` and `solution_index` are mutually exclusive:
- A static fit has `parallax_branch=NONE` and `solution_index=None`
- A ground-only (n_loc=1) parallax fit has `parallax_branch` set to `U0_PLUS` or `U0_MINUS`, and `solution_index=None`
- A multi-location (n_loc>1) parallax fit has `parallax_branch=NONE` and `solution_index` set to a non-negative integer

**R1.1.3** If both `parallax_branch != NONE` and `solution_index is not None` are set simultaneously, `__post_init__` must raise `ValueError` with a message explaining the mutual exclusivity rule.

**R1.1.4** `solution_index` values are assigned sequentially (0, 1, 2, ...) after duplicate solutions are merged. Index 0 is the best-fitting solution (lowest chi2).

**R1.1.5** All existing validation must be preserved (e.g., point lens cannot have orbital motion).

**R1.1.6** `FitKey` must remain hashable and usable as a dictionary key after these changes.

#### Interface

```python
@dataclass(frozen=True)
class FitKey:
    lens_type: LensType
    source_type: SourceType
    parallax_branch: ParallaxBranch
    lens_orb_motion: LensOrbMotion
    locations_used: Optional[str] = None
    solution_index: Optional[int] = None  # NEW

    def __post_init__(self) -> None: ...
```

---

### 1.2 `FitRecord`

#### Background
`FitRecord` stores the result of a single fit. Currently it stores `renorm_factors` (the renormalization state at fit time) but does not store which datasets were used. This makes it impossible to determine whether a fit is stale when datasets are added or renormalization changes. Two new fields support staleness checking (`dataset_labels`, `location_labels`), one new field tracks whether a fit needs to be rerun (`needs_refit`), and a new method `is_stale()` encapsulates the staleness logic.

#### Requirements

**R1.2.1** `FitRecord` must have a `dataset_labels` field of type `Optional[frozenset]`, defaulting to `None`. It stores the labels of all datasets used in the fit.

**R1.2.2** `FitRecord` must have a `location_labels` field of type `Optional[dict]`, defaulting to `None`. It stores a mapping of location name to `frozenset` of dataset labels for each location represented in the fit (e.g., `{'ground': frozenset(['OGLE', 'MOA'])}`).

**R1.2.3** `FitRecord` must have a `needs_refit` field of type `bool`, defaulting to `False`. It stores whether the fit has been marked as needing to be re-run. A record with `needs_refit=True` is kept in `all_fit_results` and its `params` are used as the starting point when the fit is re-run.

**R1.2.4** The valid combinations of `is_complete` and `needs_refit` are:

| `is_complete` | `needs_refit` | Meaning |
|---|---|---|
| `False` | `False` | User-provided initial guess, not yet fit |
| `True` | `False` | Valid, up-to-date fit result |
| `True` | `True` | Previously fit, now needs re-running; `params` used as seed |
| `False` | `True` | Invalid state — must not occur |

**R1.2.5** `__post_init__` or `from_full_result()` must raise `ValueError` if `is_complete=False` and `needs_refit=True` are set simultaneously.

**R1.2.6** `from_full_result()` must accept `dataset_labels`, `location_labels` as new optional parameters and store them on the resulting record. `needs_refit` must always be set to `False` by `from_full_result()`.

**R1.2.7** `FitRecord` must have an `is_stale()` method with the signature:
```python
def is_stale(
    self,
    current_renorm_factors: dict,
    current_location_groups: dict,
) -> bool: ...
```
where `current_location_groups` is a dict mapping location names to lists of `MulensData` objects.

**R1.2.8** `is_stale()` must return `False` for any record with `fixed=True`, regardless of any other conditions.

**R1.2.9** `is_stale()` must return `True` for any record with `dataset_labels=None` (legacy records without staleness metadata). R1.2.8 takes priority over R1.2.9

**R1.2.10** A fit is stale due to renormalization change if: for any label in `dataset_labels`, the value of `current_renorm_factors.get(label)` differs from `self.renorm_factors.get(label)`. This includes the case where a dataset was not renormalized at fit time but is now renormalized, and vice versa.

**R1.2.11** A fit is stale due to dataset addition if: for any location in `self.location_labels`, the set of current dataset labels for that location is a strict superset of the stored labels.

**R1.2.12** Adding datasets from a location not represented in `self.location_labels` must not cause the fit to be considered stale.

**R1.2.13** `is_stale()` must return `False` if neither R1.2.10 nor R1.2.11 conditions are met.

**R1.2.14** `run_fit_if_needed()` must treat a record with `needs_refit=True` as requiring a new fit, using the existing `params` as the starting point. The guard condition must be updated from:
```python
if record is not None and (record.fixed or record.is_complete):
    return record
```
to:
```python
if record is not None and (record.fixed or (record.is_complete and not record.needs_refit)):
    return record
```

**R1.2.15** - stale if **any** dataset label in `self.dataset_labels` is **missing** from current datasets (globally or in its location group)

**R1.2.16** - `FitRecord.is_stale()` must return `False` when `is_complete=False`,
#### Interface

```python
@dataclass
class FitRecord:
    model_key: FitKey
    params: dict
    sigmas: dict = None
    renorm_factors: dict = None
    dataset_labels: Optional[frozenset] = None       # NEW
    location_labels: Optional[dict] = None            # NEW
    needs_refit: bool = False                         # NEW
    full_result: object = None
    fixed: bool = False
    is_complete: bool = False

    def __post_init__(self) -> None: ...

    @classmethod
    def from_full_result(
        cls,
        model_key: FitKey,
        full_result,
        renorm_factors: dict = None,
        dataset_labels: frozenset = None,    # NEW
        location_labels: dict = None,        # NEW
        fixed: bool = False,
    ) -> 'FitRecord': ...

    def is_stale(
        self,
        current_renorm_factors: dict,
        current_location_groups: dict,
    ) -> bool: ...

    # Existing methods unchanged:
    def to_dataframe(self) -> pd.DataFrame: ...
    def chi2(self) -> Optional[float]: ...
    def __repr__(self) -> str: ...
```

---

### 1.3 Tests

```python
# test_fit_key.py

def test_solution_index_valid():
    """R1.1.1, R1.1.2: Multi-location fit with solution_index is valid."""

def test_parallax_branch_valid():
    """R1.1.2: Ground-only fit with parallax_branch is valid."""

def test_static_fit_valid():
    """R1.1.2: Static fit with neither field set is valid."""

def test_mutual_exclusivity_raises():
    """R1.1.3: Both parallax_branch and solution_index set raises ValueError."""

def test_mutual_exclusivity_error_message():
    """R1.1.3: ValueError message explains the mutual exclusivity rule."""

def test_solution_index_zero_is_best_fit():
    """R1.1.4: solution_index=0 corresponds to lowest chi2 after merging."""

def test_point_lens_orbital_motion_raises():
    """R1.1.5: Existing validation preserved."""

def test_fitkey_hashable():
    """R1.1.6: FitKey with solution_index is usable as a dict key."""

def test_fitkey_equality():
    """R1.1.6: Two FitKeys with same fields are equal."""


# test_fit_record.py

def test_needs_refit_defaults_false():
    """R1.2.3: needs_refit defaults to False."""

def test_incomplete_needs_refit_raises():
    """R1.2.5: is_complete=False and needs_refit=True raises ValueError."""

def test_from_full_result_sets_needs_refit_false():
    """R1.2.6: from_full_result() always sets needs_refit=False."""

def test_from_full_result_stores_dataset_labels():
    """R1.2.6: from_full_result() stores dataset_labels correctly."""

def test_from_full_result_stores_location_labels():
    """R1.2.6: from_full_result() stores location_labels correctly."""

def test_from_full_result_none_dataset_labels():
    """R1.2.6: from_full_result() with no dataset_labels stores None."""

def test_run_fit_if_needed_skips_complete_not_stale():
    """R1.2.14: Record with is_complete=True, needs_refit=False is reused."""

def test_run_fit_if_needed_reruns_when_needs_refit():
    """R1.2.14: Record with needs_refit=True triggers refit."""

def test_run_fit_if_needed_uses_params_as_seed():
    """R1.2.14: Refit uses existing params as starting point."""


# test_fit_record_staleness.py

def test_not_stale_when_nothing_changed():
    """R1.2.13: No staleness when renorm and datasets unchanged."""

def test_fixed_record_never_stale():
    """R1.2.8: Fixed records are never stale regardless of state."""

def test_legacy_record_always_stale():
    """R1.2.9: Records with dataset_labels=None are always stale."""

def test_stale_when_renorm_factor_changed():
    """R1.2.10: Renorm factor change makes fit stale."""

def test_stale_when_dataset_newly_renormalized():
    """R1.2.10: Dataset renormalized after fit was run makes fit stale."""

def test_stale_when_renorm_factor_removed():
    """R1.2.10: Renorm factor present at fit time but now absent makes fit stale."""

def test_stale_when_same_location_dataset_added():
    """R1.2.11: New dataset at same location makes fit stale."""

def test_not_stale_when_different_location_dataset_added():
    """R1.2.12: New dataset at different location does not make fit stale."""

def test_stale_when_dataset_removed():
    """R1.2.15: Removing a dataset after fit was run makes fit stale."""
```

---

## Section 2: Stage Tracker — Specification

---

### 2.1 Background

Currently `MMEXOFASTFitter` uses `_infer_workflow_state()` to reconstruct where the workflow is by inspecting existing fit results. This is fragile — it can misidentify the current stage when fit results are ambiguous or when users provide partial initial state.

The new design uses an explicit stage tracker stored in the restart state. The stage is set at each workflow checkpoint and is the authoritative record of where the workflow is. When the stage tracker is absent (e.g., user-provided initial results without a restart file), the system falls back to inference to set the initial stage.

---

### 2.2 Stage Definitions

A stage is either a string (for stages that don't depend on location) or a tuple (for stages that track per-location progress).

#### Point-lens stages (in order):

| Stage | Meaning |
|---|---|
| `"start"` | No work done |
| `"event_search_done"` | EF grid search complete |
| `"static_fit_done"` | Static PSPL (and FSPL if `finite_source`) complete for primary location |
| `"pl_fit_done"` | Parallax fits complete for primary location |
| `"primary_renorm_done"` | Primary location fully renormalized and refit |
| `("alt_loc_added", [loc1, ...])` | Fits complete for listed alternate locations, not yet renormalized |
| `("alt_loc_renorm_done", [loc1, ...])` | Listed alternate locations renormalized and refit |
| `"complete"` | All locations fit and renormalized |

#### Additional binary-lens stages (inserted after `"primary_renorm_done"`):

| Stage | Meaning |
|---|---|
| `"anomaly_search_done"` | AnomalyFinder grid search complete |
| `"binary_fit_done"` | Binary lens model fit complete |
| `"post_binary_renorm_done"` | Post-binary renormalization check complete |

The alt-location stages follow `"post_binary_renorm_done"` in the binary workflow and `"primary_renorm_done"` in the point-lens workflow.

---

### 2.3 Requirements

**R2.1** The stage tracker must be stored as a single attribute `self.workflow_stage` on `MMEXOFASTFitter`.

**R2.2** `workflow_stage` must be included in the state dict returned by `_get_state()` and restored by `_restore_state()`.

**R2.3** The stage must be updated at each workflow checkpoint by calling `_set_stage(stage)`. This method must log the new stage and save the restart state.

**R2.4** If `workflow_stage` is `None` on initialization (no restart file, no `restart_from` parameter, no user-provided state), it must be set to `"start"`.

**R2.5** If a `restart_from` parameter is provided to `__init__`, it overrides any stage from the restart file. Valid values are any stage string or tuple from Section 2.2. An invalid value must raise `ValueError` at `__init__` time listing the valid options.

**R2.6** If a restart file is loaded but `restart_from` is not provided, the stage from the restart file is used directly without inference.

**R2.7** If no restart file is provided but user-supplied state is present (`initial_results`, `best_ef_grid_point`, etc.), `_infer_stage_from_results()` must be called to set the initial stage.

**R2.8** `_infer_stage_from_results()` must determine the stage by inspecting available state in priority order:

1. If `all_fit_results` contains a complete binary fit → `"binary_fit_done"` (or later)
2. If best_af_grid_point is not None → 'anomaly_search_done' 
3. If primary location fits exist and are fully renormalized → `"primary_renorm_done"`
4. If primary location parallax fits exist (complete or incomplete) → `"pl_fit_done"`
5. If primary location static fits exist (complete or incomplete) → `"static_fit_done"`
6. If `best_ef_grid_point is not None` → `"event_search_done"`
7. Otherwise → `"start"`

**R2.9** For alt-location stages, `_infer_stage_from_results()` must additionally inspect `renorm_factors` and `all_fit_results` to determine which alternate locations have been added and/or renormalized.

**R2.10** The `alt_location_order` parameter (list of location names, optional) controls the order in which alternate locations are processed. If not provided, locations are ordered by time coverage (longest first), excluding the primary location.

**R2.11** The tuple form of alt-location stages `("alt_loc_added", [loc1, ...])` must use a list that reflects the ordered subset of locations processed so far, consistent with `alt_location_order`.

**R2.12** `_set_stage(stage)` must validate that the new stage is a legal forward progression from the current stage. Attempting to set a stage that is earlier in the sequence must raise `ValueError`, **with the following exception:** when new locations are detected at the start of `fit()` (R3.1.8), the stage may be reset backward to `"primary_renorm_done"` (point-lens) or `"post_binary_renorm_done"` (binary) to allow the alt-location sub-workflow to execute. This exception must be explicitly signaled — for example via a `force=True` parameter or a dedicated internal method — so the bypass is intentional and traceable.

**R2.13** `_set_stage(stage)` must write the new stage to the log.

**R2.14** After `_set_stage()`, a restart state save must be triggered automatically.

**R2.15** When `fit()` is called and `workflow_stage` is `"complete"`, the system must check whether any settings have changed since the last run and respond as follows:
- `fit_type` changed (PL→binary): reset stage to `"primary_renorm_done"`, proceed with binary stages
- `finite_source` changed: reset stage to `"static_fit_done"`, mark all existing fits `needs_refit=True`
- `parallax_grid` changed to `True`: run final parallax grid only, do not reset stage
- No settings changed and no stale fits: treat as no-op, log that workflow is already complete
- renormalize_errors changed from False to True: reset stage to "pl_fit_done", mark all existing fits needs_refit=True, proceed with renormalization from that stage
- finite_source change has nuanced behavior depending on whether a binary fit exists (you said “re-entry depends on fit-type and current state”)

---

### 2.4 Interface

```python
class MMEXOFASTFitter:

    def __init__(
        self,
        ...,
        restart_from: Optional[str] = None,
        alt_location_order: Optional[list] = None,
        ...
    ): ...

    def _set_stage(self, stage) -> None:
        """Set workflow_stage, log the transition, and save restart state."""
        ...

    def _infer_stage_from_results(self) -> None:
        """
        Set workflow_stage by inspecting current fit results and state.
        Called when user provides initial state without a restart file.
        """
        ...

    def _get_ordered_alt_locations(self) -> list:
        """
        Return alternate locations in processing order.
        Uses alt_location_order if provided, otherwise orders by time coverage.
        """
        ...

    def _is_valid_stage_progression(self, new_stage) -> bool:
        """
        Return True if new_stage is a valid forward progression
        from the current workflow_stage.
        """
        ...
```

---

### 2.5 Tests

```python
# test_stage_tracker.py

def test_initial_stage_is_start():
    """R2.4: Stage is 'start' when no restart file or initial state provided."""

def test_restart_from_overrides_restart_file():
    """R2.5: restart_from parameter overrides stage from restart file."""

def test_invalid_restart_from_raises():
    """R2.5: Invalid restart_from value raises ValueError at __init__ time."""

def test_invalid_restart_from_error_lists_valid_options():
    """R2.5: ValueError message lists valid stage options."""

def test_restart_file_stage_used_without_restart_from():
    """R2.6: Stage from restart file is used when restart_from not provided."""

def test_infer_stage_with_no_state_returns_start():
    """R2.8 (rule 7): No state → inferred stage is 'start'."""

def test_infer_stage_with_ef_grid_point():
    """R2.8 (rule 6): best_ef_grid_point present → 'event_search_done'."""

def test_infer_stage_with_static_fits():
    """R2.8 (rule 5): Static fits present → 'static_fit_done'."""

def test_infer_stage_with_parallax_fits():
    """R2.8 (rule 4): Parallax fits present → 'pl_fit_done'."""

def test_infer_stage_with_renormalized_primary():
    """R2.8 (rule 3): Primary location renormalized → 'primary_renorm_done'."""

def test_infer_stage_with_binary_fit():
    """R2.8 (rule 1): Binary fit present → 'binary_fit_done'."""

def test_infer_stage_alt_location_added():
    """R2.9: Alternate location fits present → correct alt_loc_added tuple."""

def test_infer_stage_alt_location_renormed():
    """R2.9: Alternate location renormalized → correct alt_loc_renorm_done tuple."""

def test_alt_location_order_respected():
    """R2.10: alt_location_order parameter controls processing order."""

def test_alt_location_order_default_by_coverage():
    """R2.10: Default order is by time coverage, longest first."""

def test_alt_loc_tuple_reflects_partial_progress():
    """R2.11: Tuple list reflects only locations processed so far."""

def test_set_stage_updates_workflow_stage():
    """R2.1, R2.3: _set_stage() updates self.workflow_stage."""

def test_set_stage_logs_transition():
    """R2.13: _set_stage() writes new stage to log."""

def test_set_stage_saves_restart_state():
    """R2.14: _set_stage() triggers restart state save."""

def test_set_stage_backward_progression_raises():
    """R2.12: Setting an earlier stage raises ValueError."""

def test_stage_saved_in_state():
    """R2.2: workflow_stage is included in _get_state() output."""

def test_stage_restored_from_state():
    """R2.2: workflow_stage is restored by _restore_state()."""

def test_complete_stage_fit_type_change():
    """R2.15: fit_type change at 'complete' resets to primary_renorm_done."""

def test_complete_stage_finite_source_change():
    """R2.15: finite_source change at 'complete' resets to static_fit_done."""

def test_complete_stage_parallax_grid_added():
    """R2.15: parallax_grid=True added at 'complete' runs grid only."""

def test_complete_stage_no_changes_is_noop():
    """R2.15: No changes at 'complete' with no stale fits is a no-op."""

def test_complete_stage_renormalize_errors_enabled():
    """R2.15: renormalize_errors changed to True at 'complete' resets to pl_fit_done."""

```

---

## Section 3: Core Logic — Specification

---

### 3.1 Staleness Checking in `MMEXOFASTFitter`

#### Background
`FitRecord.is_stale()` checks staleness for a single record. `MMEXOFASTFitter` needs higher-level methods that iterate over all records, identify stale ones, mark them for re-running, and reset the workflow stage accordingly. These are called at the start of every `fit()` invocation.

#### Requirements

**R3.1.1** `MMEXOFASTFitter` must have a method `_get_stale_keys()` that returns a list of `FitKey` objects for all records in `all_fit_results` where `is_stale()` returns `True`.

**R3.1.2** `_get_stale_keys()` must pass the current `self.renorm_factors` and `self.location_groups` to each record's `is_stale()` method.

**R3.1.3** Fixed records must never appear in the stale keys list (enforced by `FitRecord.is_stale()`).

**R3.1.4** `MMEXOFASTFitter` must have a method `_mark_stale_fits()` that sets `needs_refit=True` on all stale records in `all_fit_results`. Records are not removed — their `params` are preserved as seeds for re-running.

**R3.1.5** After marking stale fits, `_mark_stale_fits()` must reset `workflow_stage` to the earliest stage implied by the marked fits:
- If any static fit was marked → reset to `"event_search_done"`
- If only parallax fits were marked → reset to `"static_fit_done"`
- If only binary fits were marked → reset to `"primary_renorm_done"`
- If only alt-location fits were marked → reset to ("alt_loc_added", [locs_processed_so_far_excluding_the_stale_location])

**R3.1.6** `_mark_stale_fits()` must log which fits were marked and what stage was reset to.

**R3.1.7** `_mark_stale_fits()` must be called at the start of every `fit()` invocation, before any workflow steps run.

**R3.1.8** At the start of `fit()`, after `_mark_stale_fits()` and the R2.15 settings-change check, the fitter must determine whether any datasets have been added from a previously unrepresented location by comparing `self.location_groups` against locations in existing `FitRecord.location_labels`. If new locations are detected:
- Existing fits are NOT marked `needs_refit=True`
- The stage is reset to the pre-alt-location entry point: `"primary_renorm_done"` for `fit_type='point lens'`, `"post_binary_renorm_done"` for `fit_type='binary lens'`. This reset bypasses the forward-progression check per R2.12.
- The single-pass different-location workflow (R3.2.2) is triggered for those locations.
- For `fit_type="binary lens"` and `workflow_stage < "post_binary_renorm_done"`, newly detected non-primary locations are not processed yet — they will be handled after the binary stage.

---

### 3.2 Dataset Addition Behavior

#### Same-location dataset addition (two-pass):

**R3.2.1** When new datasets are detected from a location already represented in existing fit records, the two-pass workflow must execute:

**Pass 1:**
1. New datasets are loaded fresh with no renorm factor
2. fits have been marked needs_refit=True by the initial _mark_stale_fits() call
3. Stage is reset per R3.1.5
4. All `needs_refit=True` fits are refit using existing `params` as seeds, with all same-location datasets (old renorm'd + new unrenorm'd)
5. Restart state is saved

**Pass 2:**
6. Best current model selected via `_select_reference_model()`
7. Outlier removal and errfac calculation run on new (unrenormalized) datasets only
8. New errfacs applied, datasets recreated with scaled errors
9. All fits using those datasets marked `needs_refit=True`
10. All `needs_refit=True` fits refit using current `params` as seeds
11. *(if `parallax_grid=True`)* Final parallax grid run once, after Pass 2 only
12. Restart state saved

#### Different-location dataset addition (single-pass):

**R3.2.2** When new datasets are detected from a location not represented in any existing fit record, the single-pass workflow must execute:

1. New datasets loaded fresh
2. Existing fits are NOT marked `needs_refit=True`. The stage is reset to the pre-alt-location entry point per R3.1.8.
3. Static PSPL fit run with all current datasets (primary + new location)
4. Coarse parallax grid always run (required to find initial multi-location solutions)
5. Minima optimized, duplicate solutions merged
6. New multi-location parallax fits created, seeded from coarse grid solutions
7. Best multi-location fit selected via `_select_reference_model()`
8. Outlier removal and errfac calculation on new datasets
9. New errfacs applied, datasets recreated with scaled errors
10. Multi-location parallax fits marked `needs_refit=True`
11. All `needs_refit=True` fits refit using current `params` as seeds
12. *(if `parallax_grid=True`)* Final parallax grid run
13. Restart state saved


---

### 3.3 Solution Merging

#### Background
When multiple minima are found in the parallax grid and optimized, some converge to the same solution. Duplicate solutions must be detected and merged before being stored. The user must be informed of the merging in the log. The same logic applies to binary lens solutions.

#### Requirements

**R3.3.1** `MMEXOFASTFitter` must have a method `_merge_duplicate_solutions(solutions)` that accepts a list of `FitRecord` objects and returns a deduplicated list.

**R3.3.2** Two solutions are considered duplicates if all parameters in `parameters_to_fit` agree within their respective 1-sigma uncertainties. Specifically, for each parameter `p`: `|params_a[p] - params_b[p]| <= max(sigmas_a[p], sigmas_b[p])`.

**R3.3.3** If a solution has no `sigmas`, it must not be merged with any other solution. A warning must be logged.

**R3.3.4** Among a group of duplicate solutions, the one with the lowest chi2 is kept. The others are discarded.

**R3.3.5** The method must log: the original number of solutions, the number of unique solutions after merging, and for each merged group the parameters of all discarded solutions.

**R3.3.6** After merging, surviving solutions are assigned sequential `solution_index` values (0, 1, 2, ...) ordered by chi2 ascending (index 0 = best fit).

**R3.3.7** The same merging logic applies to both point-lens parallax solutions and binary lens solutions.

---

### 3.4 `_select_reference_model()`

#### Background
Several methods currently hardcode `_select_preferred_static_point_lens_model()` as the reference for grid searches and renormalization. The correct reference depends on what fits exist. This logic must be centralized.

#### Requirements

**R3.4.1** `MMEXOFASTFitter` must have a method `_select_reference_model()` that returns the best available `FitRecord` for use as a reference model.

**R3.4.2** Selection priority (highest to lowest):
1. Best complete binary lens fit (lowest chi2 among all binary fits)
2. Best complete parallax point-lens fit (lowest chi2 among all parallax fits matching current n_loc)
3. Best complete static point-lens fit (FSPL preferred over PSPL if chi2 improvement exceeds threshold)

Note: "Matching current n_loc" is a hard constraint. _select_reference_model() must not be used for seeding in contexts where no same-n_loc fits exist yet. In those cases, callers must use _find_related_fit() instead.

**R3.4.3** If no complete fits exist, `_select_reference_model()` must raise `RuntimeError` with a descriptive message.

**R3.4.4** All existing calls to `_select_preferred_static_point_lens_model()` in methods that use a reference model for grid searches or renormalization must be replaced with calls to `_select_reference_model()`.

**R3.4.5** `_select_preferred_static_point_lens_model()` must be retained as a separate method for cases where a static model is explicitly required (e.g., seeding parallax fits).

**R3.4.6** `MMEXOFASTFitter` must have a method `_select_preferred_point_lens()` that returns the best available point-lens `FitRecord`. Selection priority:
1. Best complete parallax point-lens fit (lowest chi2 among all parallax fits matching current n_loc)
2. Best complete static point-lens fit (FSPL preferred over PSPL if chi2 improvement exceeds threshold)

If no complete point-lens fits exist, `_select_preferred_point_lens()` must raise `RuntimeError` with a descriptive message.

**R3.4.7** All existing calls to `_select_best_static_pspl()` in `get_anomaly_lc_params()` must be replaced with calls to `_select_preferred_point_lens()`.

**R3.4.8** `_select_best_static_pspl()` must be retained for cases where a static PSPL model is explicitly required.

#### Updated Interface (Section 3.4)

```python
def _select_reference_model(self) -> FitRecord:
    """Return best available fit: binary > parallax > static."""
    ...

def _select_preferred_point_lens(self) -> FitRecord:
    """Return best available point-lens fit: parallax > static.
    Raises RuntimeError if no complete point-lens fits exist."""
    ...

def _select_preferred_static_point_lens_model(self) -> FitRecord:
    """Return best static fit (FSPL preferred over PSPL if chi2 threshold met).
    Used for seeding parallax fits."""
    ...

def _select_best_static_pspl(self) -> FitRecord:
    """Return best static PSPL fit. Retained for cases requiring PSPL specifically."""
    ...
```

---

### 3.5 `_needs_renormalization()`

#### Background
In the binary workflow, after fitting the binary model, the code must check whether the binary model warrants a second renormalization cycle. The threshold is deferred but the architecture must support adding it.

#### Requirements

**R3.5.1** `MMEXOFASTFitter` must have a method `_needs_renormalization(reference_model, datasets)` that returns `bool`.

**R3.5.2** The method must compute `errfac = sqrt(chi2 / dof)` for each dataset relative to `reference_model`.

**R3.5.3** The method must return `True` if any dataset's `errfac` deviates from 1.0 by more than `RENORM_THRESHOLD`.

**R3.5.4** `RENORM_THRESHOLD` must be a class-level constant defaulting to `None` (disabled). When `None`, the method always returns `False`.

**R3.5.5** The method must log the computed errfac for each dataset regardless of return value.

**R3.5.6** When returning `True`, the method must log which datasets exceeded the threshold and by how much.

---

### 3.6 Ephemeris Bounds Check in `_get_space_u0_sign()`

#### Background
The crash in the original code occurred because `minimize_scalar` was called with bounds derived from `t_0 ± 2*t_E`, but the optimized `t_0` had shifted far enough that the lower bound fell outside the Spitzer ephemeris window.

#### Requirements

**R3.6.1** Before calling `minimize_scalar`, `_get_space_u0_sign()` must read the time bounds of the satellite ephemeris file and clamp the search bounds to the intersection of `(t_0 - 2*t_E, t_0 + 2*t_E)` and the ephemeris time range.

**R3.6.2** If the intersection is empty, `_get_space_u0_sign()` must raise `ValueError` stating both the ephemeris range and the requested range.

**R3.6.3** If the search bounds were clamped, a warning must be logged stating the original and clamped bounds.

---

### 3.7 Unrenormalized Dataset Warning

#### Requirements

**R3.7.1** `MMEXOFASTFitter` must have a method `_check_renorm_completeness()` that returns a list of dataset labels not present in `self.renorm_factors`.

**R3.7.2** `_check_renorm_completeness()` must be called:
- At the end of `fit()`, writing any warning to the log
- Before `_output_latex_table()`
- Before `initialize_exozippy()`

**R3.7.3** The warning message must list the specific dataset labels that have not been renormalized.

**R3.7.4** The warning is only issued when renormalization has been partially applied — i.e., some but not all datasets are in `renorm_factors`. If `renorm_factors` is empty or contains all dataset labels, no warning is issued.

---

### 3.8 Methods Requiring Updates for FitKey Changes
The following existing methods use parallax branch enums (U0_PP, U0_PM, U0_MP, U0_MM) that no longer exist for multi-location fits in the new design. Each must be updated.

R3.8.1 _iter_parallax_point_lens_keys() currently yields FitKey objects with U0_PP/U0_PM/U0_MP/U0_MM branches for n_loc>1. It must be updated to instead yield FitKey objects with solution_index values for existing solutions when n_loc>1. If no solutions exist yet (e.g., before the coarse grid has run), it must yield nothing for n_loc>1.

R3.8.2 _parallax_fits_match_n_loc() currently checks for the presence of PP/PM/MP/MM branches to determine if existing parallax fits match the current n_loc. It must be updated to check for the presence of solution_index-based fits when n_loc>1.

R3.8.3 _find_related_fit() currently matches on locations_used to find fits with the same model type but different location coverage. It must be updated to handle the case where multi-location fits use solution_index rather than parallax_branch. Specifically:

When searching for a related fit to seed a new multi-location fit and same-n_loc fits exist, solution_index=0 (best existing solution) is the preferred seed regardless of locations_used.
When no multi-location fits exist at any n_loc (e.g., the first alternate location is being added), _find_related_fit() must fall back to _select_preferred_point_lens() to obtain the best available point-lens seed.

### 3.9 Deprecated Methods
R3.9.1 _infer_workflow_state() is replaced by _infer_stage_from_results() (R2.8) and must be removed. Any code that calls _infer_workflow_state() must be updated to use the stage tracker (self.workflow_stage) directly.

R3.9.2 _log_workflow_state() which depends on _infer_workflow_state() must also be removed. Workflow state logging is now handled by _set_stage() (R2.13) and _mark_stale_fits() (R3.1.6).

### 3.10 run_fit_if_needed() Update
R3.10.1 run_fit_if_needed() must treat a record with needs_refit=True as requiring a new fit, using the existing params as the starting point. The guard condition must be updated from:

if record is not None and (record.fixed or record.is_complete):
    return record

to:

if record is not None and (record.fixed or (record.is_complete and not record.needs_refit)):
    return record

Add corresponding tests in test_run_fit_if_needed.py:

def test_run_fit_if_needed_skips_complete_not_stale():
    """R3.10.1: Record with is_complete=True, needs_refit=False is reused."""

def test_run_fit_if_needed_reruns_when_needs_refit():
    """R3.10.1: Record with needs_refit=True triggers refit."""

def test_run_fit_if_needed_uses_params_as_seed():
    """R3.10.1: Refit uses existing params as starting point."""

### 3.10 Tests

```python
# test_staleness_in_fitter.py

def test_get_stale_keys_returns_stale_records():
    """R3.1.1, R3.1.2: Stale records are identified correctly."""

def test_get_stale_keys_excludes_fixed():
    """R3.1.3: Fixed records never appear in stale keys."""

def test_get_stale_keys_empty_when_nothing_stale():
    """R3.1.1: Returns empty list when no fits are stale."""

def test_mark_stale_fits_sets_needs_refit():
    """R3.1.4: Stale fits have needs_refit set to True."""

def test_mark_stale_fits_preserves_params():
    """R3.1.4: Marked fits remain in all_fit_results with params intact."""

def test_mark_stale_fits_resets_stage_static_marked():
    """R3.1.5: Stage resets to event_search_done when static fit marked."""

def test_mark_stale_fits_resets_stage_parallax_only():
    """R3.1.5: Stage resets to static_fit_done when only parallax fits marked."""

def test_mark_stale_fits_resets_stage_binary_only():
    """R3.1.5: Stage resets to primary_renorm_done when only binary fits marked."""

def test_mark_stale_fits_logs_marked_fits():
    """R3.1.6: Marked fits and new stage are logged."""

def test_mark_stale_fits_called_at_start_of_fit():
    """R3.1.7: _mark_stale_fits() called before any workflow steps."""

def test_new_location_triggers_different_location_workflow():
    """R3.1.8: New location detected triggers single-pass workflow."""


# test_dataset_addition.py

def test_same_location_addition_pass1_marks_fits_stale():
    """R3.2.1: Pass 1 marks same-location fits needs_refit=True."""

def test_same_location_addition_pass1_refits_with_all_datasets():
    """R3.2.1: Pass 1 refits using old renorm'd + new unrenorm'd datasets."""

def test_same_location_addition_pass2_renormalizes_new_only():
    """R3.2.1: Pass 2 renormalizes only previously unrenormalized datasets."""

def test_same_location_addition_parallax_grid_after_pass2_only():
    """R3.2.1: Parallax grid runs after Pass 2, not after Pass 1."""

def test_different_location_coarse_grid_always_runs():
    """R3.2.2: Coarse grid always runs for different-location addition."""

def test_different_location_existing_fits_not_marked_stale():
    """R3.2.2: Existing fits not marked needs_refit for different-location addition."""

def test_different_location_final_grid_only_if_parallax_grid_true():
    """R3.2.2: Final grid only runs if parallax_grid=True."""


# test_solution_merging.py

def test_merge_identical_solutions():
    """R3.3.1, R3.3.4: Identical solutions are merged, lowest chi2 kept."""

def test_merge_solutions_within_sigma():
    """R3.3.2: Solutions within 1-sigma are merged."""

def test_no_merge_outside_sigma():
    """R3.3.2: Solutions outside 1-sigma are not merged."""

def test_no_merge_without_sigmas():
    """R3.3.3: Solutions without sigmas are not merged, warning logged."""

def test_merge_logs_original_and_final_counts():
    """R3.3.5: Log contains original and final solution counts."""

def test_merge_logs_discarded_solutions():
    """R3.3.5: Log contains parameters of each discarded solution."""

def test_solution_index_assigned_by_chi2():
    """R3.3.6: Index 0 assigned to lowest chi2 after merging."""

def test_merge_applies_to_binary_solutions():
    """R3.3.7: Merging works for binary lens solutions."""


# test_select_reference_model.py

def test_binary_preferred_over_parallax():
    """R3.4.2: Binary fit preferred over parallax when both exist."""

def test_parallax_preferred_over_static():
    """R3.4.2: Parallax fit preferred over static when both exist."""

def test_fspl_preferred_over_pspl_if_significantly_better():
    """R3.4.2: FSPL preferred over PSPL if chi2 improvement exceeds threshold."""

def test_raises_when_no_complete_fits():
    """R3.4.3: RuntimeError raised when no complete fits exist."""

def test_replaces_hardcoded_static_reference_in_grid_search():
    """R3.4.4: _run_piE_grid_search uses _select_reference_model()."""

def test_replaces_hardcoded_static_reference_in_renormalization():
    """R3.4.4: Renormalization methods use _select_reference_model()."""

def test_select_preferred_point_lens_returns_parallax_over_static():
    """R3.4.6: _select_preferred_point_lens() returns parallax fit over static when both exist."""

def test_select_preferred_point_lens_returns_static_when_no_parallax():
    """R3.4.6: _select_preferred_point_lens() returns static fit when no parallax fits exist."""

def test_select_preferred_point_lens_raises_when_no_complete_fits():
    """R3.4.6: RuntimeError raised when no complete point-lens fits exist."""

def test_get_anomaly_lc_params_uses_select_preferred_point_lens():
    """R3.4.7: get_anomaly_lc_params() uses _select_preferred_point_lens() not _select_best_static_pspl()."""
    
# test_needs_renormalization.py

def test_returns_false_when_threshold_is_none():
    """R3.5.4: Returns False when RENORM_THRESHOLD is None."""

def test_returns_false_when_errfac_near_one():
    """R3.5.3: Returns False when all errfacs within threshold."""

def test_returns_true_when_errfac_exceeds_threshold():
    """R3.5.3: Returns True when any dataset errfac exceeds threshold."""

def test_logs_errfac_for_all_datasets():
    """R3.5.5: errfac logged for each dataset regardless of result."""

def test_logs_exceeding_datasets_when_true():
    """R3.5.6: Datasets exceeding threshold logged when True returned."""


# test_ephemeris_bounds.py

def test_bounds_clamped_to_ephemeris_range():
    """R3.6.1: Search bounds clamped to ephemeris time range."""

def test_raises_when_window_outside_ephemeris():
    """R3.6.2: ValueError raised when entire window outside ephemeris."""

def test_error_message_states_both_ranges():
    """R3.6.2: Error message states both ephemeris and requested ranges."""

def test_warning_logged_when_bounds_clamped():
    """R3.6.3: Warning logged when bounds are clamped."""

def test_no_warning_when_bounds_not_clamped():
    """R3.6.3: No warning when bounds within ephemeris range."""


# test_renorm_completeness.py

def test_no_warning_when_all_renormalized():
    """R3.7.4: No warning when all datasets renormalized."""

def test_no_warning_when_none_renormalized():
    """R3.7.4: No warning when renorm_factors is empty."""

def test_warning_when_partially_renormalized():
    """R3.7.1, R3.7.3: Warning lists unrenormalized dataset labels."""

def test_warning_triggered_at_end_of_fit():
    """R3.7.2: Warning written to log at end of fit()."""

def test_warning_triggered_before_latex_output():
    """R3.7.2: Warning logged before _output_latex_table()."""

def test_warning_triggered_before_initialize_exozippy():
    """R3.7.2: Warning logged before initialize_exozippy()."""

# test_fitkey_method_updates.py 

def test_iter_parallax_keys_yields_solution_index_for_multiloc():
    """R3.8.1: _iter_parallax_point_lens_keys() yields solution_index keys for n_loc>1."""

def test_iter_parallax_keys_yields_nothing_before_grid_for_multiloc():
    """R3.8.1: _iter_parallax_point_lens_keys() yields nothing for n_loc>1 before grid run."""

def test_iter_parallax_keys_yields_branches_for_single_loc():
    """R3.8.1: _iter_parallax_point_lens_keys() still yields U0_PLUS/MINUS for n_loc=1."""

def test_parallax_fits_match_nloc_uses_solution_index():
    """R3.8.2: _parallax_fits_match_n_loc() correctly identifies solution_index fits for n_loc>1."""

def test_find_related_fit_uses_best_solution_index():
    """R3.8.3: _find_related_fit() uses solution_index=0 as preferred seed
    when same-n_loc fits exist."""

def test_find_related_fit_falls_back_to_point_lens_when_no_multiloc_fits():
    """R3.8.3: _find_related_fit() falls back to _select_preferred_point_lens()
    when no multi-location fits exist."""
    
# test_renormalization.py

def test_outlier_rejection_removes_outliers():
    """Outlier rejection algorithm removes data points beyond threshold."""

def test_outlier_rejection_no_outliers():
    """Outlier rejection returns all points when none exceed threshold."""

def test_errfac_calculation_correct():
    """errfac = sqrt(chi2/dof) computed correctly."""

def test_errfac_calculation_perfect_fit():
    """errfac = 1.0 when chi2 equals dof."""

def test_dataset_recreated_with_scaled_errors():
    """Dataset errors scaled by errfac when recreated."""

def test_dataset_errors_unchanged_when_errfac_is_one():
    """Dataset errors unchanged when errfac = 1.0."""
```

---

## Section 4: `__init__` Changes — Specification

---

### 4.1 Background

Several new parameters are needed to support the stage tracker, user-provided state, and conflict detection introduced in Sections 1-3. Some existing initialization logic is tightened around config conflicts and dataset state consistency.

---

### 4.2 New Parameters

#### Requirements

**R4.1.1** The following new parameters must be added to `__init__`:

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `restart_from` | `Optional[str]` | `None` | Explicit workflow entry point |
| `alt_location_order` | `Optional[list]` | `None` | Order for processing alternate locations |
| `best_ef_grid_point` | `Optional[dict]` | `None` | User-supplied EF grid result |
| `best_af_grid_point` | `Optional[dict]` | `None` | User-supplied AF grid result |
| `renorm_factors` | `Optional[dict]` | `None` | User-supplied renormalization factors |

**R4.1.2** `best_ef_grid_point`, `best_af_grid_point`, and `renorm_factors` are state parameters, not config parameters. They must not be added to `CONFIG_KEYS`.

**R4.1.3** These parameters must be handled in `_restore_state()`, not `_merge_config()`.

---

### 4.3 State Merge Priority

**R4.2.1** For `best_ef_grid_point` and `best_af_grid_point`, user-provided value wins over restart file:
```
self.best_ef_grid_point = user_provided or saved_state.get('best_ef_grid_point')
```

**R4.2.2** For `renorm_factors`, merge is per-label with user-provided winning:
```
self.renorm_factors = {
    **saved_state.get('renorm_factors', {}),
    **(user_renorm_factors or {})
}
```

**R4.2.3** When a user-provided renorm factor differs from the saved factor for any label, the corresponding dataset must be reloaded from its source file and the new factor applied. A warning must be logged for each affected dataset.

**R4.2.4** Immediately after merging renorm_factors in __init__, if any label's factor differs from the saved value, _mark_stale_fits() must be called before __init__ completes. This ensures fits are marked needs_refit=True before any restart state is saved or fit() is called.

_mark_stale_fits() is still called at the start of fit() to catch staleness from other sources (dataset additions detected at fit time, etc.), but the renorm conflict case is handled eagerly in __init__.


---

### 4.4 Config Conflict Detection

**R4.3.1** If a restart file is provided and `coords` is also provided and differs, `__init__` must raise `ValueError` stating both values. This check must occur before any state is restored.

**R4.3.2** If `fit_type` differs from the saved value, this is a valid use case (PL→binary transition). No error is raised.

**R4.3.3** If `finite_source` differs from the saved value, this is a valid use case. No error is raised.

---

### 4.5 Dataset State Consistency

**R4.4.1** When both a restart file and new `files` are provided, datasets whose labels match the restart file must use the saved version. New datasets are loaded fresh.

**R4.4.2** When user-provided `renorm_factors` includes a factor for a dataset loaded fresh from file, that dataset must be reloaded and the factor applied. A warning must be logged.

**R4.4.3** After all dataset initialization, `_check_dataset_labels_unique()` must be called.

**R4.4.4** After all dataset initialization, `self.location_groups` cache must be invalidated.

---

### 4.6 Stage Initialization Order

**R4.5.1** Stage initialization must occur after all datasets and state have been restored.

**R4.5.2** Stage is set in this priority order:
1. If `restart_from` provided → validate and use it (R2.5)
2. Else if restart file provided `workflow_stage` → use it (R2.6)
3. Else if user-provided state present → call `_infer_stage_from_results()` (R2.7)
4. Else → set to `"start"` (R2.4)

**R4.5.3** If `restart_from` is provided alongside a restart file, `restart_from` wins on stage but all other saved state is still loaded.

---

### 4.7 Primary Location

R4.6.1 MMEXOFASTFitter must determine a single primary location during __init__ and store it as self._primary_location. The primary location is determined by the following priority chain, stopping at the first match:

Saved value from restart file: If a restart file is provided and primary_location is present in the saved config, use that value directly.
Explicit primary_location parameter: If primary_location is provided to __init__, use it.
primary_dataset parameter: If primary_dataset is provided, find the location in self.location_groups whose dataset labels include primary_dataset. Use that location.
Infer from renormalized datasets: If self.renorm_factors is non-empty, find the location whose dataset labels overlap with self.renorm_factors. Use that location.
Infer from best static fit: If a complete static fit exists, use the first location listed in the fit's locations_used field. If locations_used is 'All', skip this rule.
Fallback — longest coverage: Call _select_primary_location_by_coverage(), which returns the location with the longest total time coverage across its datasets.
R4.6.2 primary_location and primary_dataset are existing config parameters and must remain in CONFIG_KEYS.

R4.6.3 self._primary_location must be saved under 'config' in the restart file so it persists across restarts.

R4.6.4 _select_primary_location_by_coverage() must return the location with the longest total time baseline across its datasets, excluding any location already identified as an alternate location. If all locations have equal coverage, the first in self.location_groups is returned.

R4.6.5 If _primary_location cannot be determined after exhausting all six rules, __init__ must raise RuntimeError with a message listing the datasets and locations available.

### 4.8 Tests

```python
# test_init_new_parameters.py

def test_best_ef_grid_point_user_wins_over_restart():
    """R4.2.1: User-provided best_ef_grid_point overrides restart file."""

def test_best_ef_grid_point_restart_used_when_not_provided():
    """R4.2.1: Restart file value used when not user-provided."""

def test_best_af_grid_point_user_wins_over_restart():
    """R4.2.1: User-provided best_af_grid_point overrides restart file."""

def test_renorm_factors_merged_per_label():
    """R4.2.2: User renorm_factors merged per label with restart file."""

def test_renorm_factors_user_wins_per_label():
    """R4.2.2: User-provided factor wins for same label."""

def test_renorm_factors_restart_preserved_for_unlisted_labels():
    """R4.2.2: Restart file factors preserved for labels not in user dict."""

def test_renorm_conflict_reloads_dataset():
    """R4.2.3: Dataset reloaded when user renorm factor differs from saved."""

def test_renorm_conflict_logs_warning():
    """R4.2.3: Warning logged when dataset reloaded due to renorm conflict."""

def test_renorm_factors_not_in_config_keys():
    """R4.1.2: renorm_factors not in CONFIG_KEYS."""

def test_best_ef_grid_point_not_in_config_keys():
    """R4.1.2: best_ef_grid_point not in CONFIG_KEYS."""

def test_best_af_grid_point_not_in_config_keys():
    """R4.1.2: best_af_grid_point not in CONFIG_KEYS."""


# test_init_conflict_detection.py

def test_coords_conflict_raises():
    """R4.3.1: Differing coords raises ValueError."""

def test_coords_conflict_error_states_both_values():
    """R4.3.1: Error message states both saved and provided coords."""

def test_coords_conflict_raises_before_state_restored():
    """R4.3.1: ValueError raised before any state is set."""

def test_fit_type_change_does_not_raise():
    """R4.3.2: Changing fit_type does not raise."""

def test_finite_source_change_does_not_raise():
    """R4.3.3: Changing finite_source does not raise."""

def test_coords_match_does_not_raise():
    """R4.3.1: Matching coords do not raise."""


# test_init_stage_initialization.py

def test_restart_from_takes_priority_over_saved_stage():
    """R4.5.2 rule 1: restart_from wins over saved stage."""

def test_restart_from_with_restart_file_loads_state():
    """R4.5.3: restart_from wins on stage but saved state still loaded."""

def test_saved_stage_used_without_restart_from():
    """R4.5.2 rule 2: Saved stage used when restart_from not provided."""

def test_infer_called_with_user_state_no_restart():
    """R4.5.2 rule 3: _infer_stage_from_results() called with user state and no restart file."""

def test_start_stage_when_no_state():
    """R4.5.2 rule 4: Stage is 'start' when no state provided."""

def test_stage_init_after_datasets_restored():
    """R4.5.1: Stage initialization after datasets and state restored."""


# test_init_dataset_consistency.py

def test_saved_dataset_used_when_label_matches():
    """R4.4.1: Saved dataset used when label matches restart file."""

def test_fresh_dataset_loaded_when_label_not_in_restart():
    """R4.4.1: Fresh dataset loaded when label not in restart file."""

def test_fresh_dataset_renormed_when_factor_provided():
    """R4.4.2: Fresh dataset reloaded and factor applied when user renorm_factors provided."""

def test_fresh_dataset_renorm_logs_warning():
    """R4.4.2: Warning logged when fresh dataset renormalized from user factors."""

def test_dataset_labels_unique_checked():
    """R4.4.3: _check_dataset_labels_unique() called after dataset initialization."""

def test_location_groups_cache_invalidated():
    """R4.4.4: location_groups cache invalidated after dataset initialization."""

def test_primary_location_from_restart_file():
    """R4.6.1 rule 1: Primary location loaded from restart file config."""

def test_primary_location_from_explicit_parameter():
    """R4.6.1 rule 2: primary_location parameter used when provided."""

def test_primary_location_from_primary_dataset():
    """R4.6.1 rule 3: primary_dataset parameter used to infer location."""

def test_primary_location_from_renorm_factors():
    """R4.6.1 rule 4: Location with renormalized datasets used as primary."""

def test_primary_location_from_static_fit():
    """R4.6.1 rule 5: First location in best static fit's locations_used used."""

def test_primary_location_from_static_fit_skips_all():
    """R4.6.1 rule 5: locations_used='All' skipped when inferring primary."""

def test_primary_location_fallback_to_coverage():
    """R4.6.1 rule 6: Longest coverage location used as fallback."""

def test_primary_location_saved_in_config():
    """R4.6.3: _primary_location saved in restart file config."""

def test_primary_location_raises_when_undetermined():
    """R4.6.5: RuntimeError raised when primary location cannot be determined."""

def test_select_primary_location_by_coverage_returns_longest():
    """R4.6.4: _select_primary_location_by_coverage() returns longest baseline location."""

def test_select_primary_location_by_coverage_tie_returns_first():
    """R4.6.4: Equal coverage returns first location in location_groups."""
```

---

## Section 5: Workflow Definitions — Specification

---

### 5.1 Background

The `fit()` method routes to the appropriate workflow based on `fit_type` and `workflow_stage`. Each call to `fit()` executes all remaining stages from the current `workflow_stage` through `"complete"`. The user can stop at any stage, do further work (add data, change settings), and resume — the workflow picks up from where it left off. The one-touch and incremental workflows share identical stage logic; the only difference is how many times `fit()` is called and when datasets are added.

---

### 5.2 `fit()` Routing

#### Requirements

**R5.1.1** At the start of every `fit()` call, before any workflow steps execute, the following must occur in this order:
3. `_mark_stale_fits()` must be called (R3.1.7)
4. If `workflow_stage` is `"complete"`, the R2.15 settings-change check must be performed. Stage and fit records may be updated as a result before any further steps run.
5. New locations must be detected and the appropriate dataset addition workflow triggered, including stage reset if needed (R3.1.8)
6. 
**R5.1.2** `fit()` must route to `fit_point_lens()` when `fit_type='point lens'`.

**R5.1.3** `fit()` must route to `fit_binary_lens()` when `fit_type='binary lens'`.

**R5.1.4** Both `fit_point_lens()` and `fit_binary_lens()` must execute only the stages that remain from the current `workflow_stage` — they must not re-execute already-completed stages.

**R5.1.5** At the end of every `fit()` call, after all workflow steps complete:
1. `_check_renorm_completeness()` must be called and any warning written to the log (R3.7.2)
2. `_output_latex_table()` must be called

**R5.1.6** If `fit_type` is `None`, `fit()` must raise `ValueError` before executing any steps.

---

### 5.3 Point-Lens Workflow

The point-lens workflow covers all stages from `"start"` through `"complete"`. For the binary workflow, the stages through `"primary_renorm_done"` are shared and executed by `fit_point_lens()` before binary-specific stages begin.

#### Requirements

**R5.2.1** `"start"` → `"event_search_done"`

Actions:
1. Run EF grid search on primary location datasets
2. Store `best_ef_grid_point`
3. Call `_set_stage("event_search_done")`

Skip condition: if `workflow_stage` is already at or past `"event_search_done"`, skip this step.

**R5.2.2** `"event_search_done"` → `"static_fit_done"`

Actions:
1. Estimate initial PSPL parameters from `best_ef_grid_point`
2. Run `run_fit_if_needed()` for static PSPL using primary location datasets
3. If `finite_source=True`: run `run_fit_if_needed()` for static FSPL using primary location datasets, seeded from PSPL params
4. Call `_set_stage("static_fit_done")`

Skip condition: if `workflow_stage` is already at or past `"static_fit_done"`, skip.

**R5.2.3** `"static_fit_done"` → `"pl_fit_done"`

Actions:
1. Run `run_fit_if_needed()` for PAR_u0+ and PAR_u0− using primary location datasets
2. Seed from `_select_preferred_static_point_lens_model()` (static model explicitly required here)
3. Call `_set_stage("pl_fit_done")`

Skip condition: if `workflow_stage` is already at or past `"pl_fit_done"`, skip.

**R5.2.4** `"pl_fit_done"` → `"primary_renorm_done"`

Actions:
1. If `renormalize_errors=True`:
   a. Select reference model via `_select_reference_model()`
   b. Run outlier removal and errfac calculation on all primary location datasets not yet in `renorm_factors`
   c. Apply errfacs, recreate datasets with scaled errors
   d. Mark all primary location fits `needs_refit=True`
   e. Refit all `needs_refit=True` fits using current `params` as seeds
   f. If `parallax_grid=True`: run final parallax grid
2. Call `_set_stage("primary_renorm_done")`

Skip condition: if `workflow_stage` is already at or past `"primary_renorm_done"`, skip.

**R5.2.5** `"primary_renorm_done"` → alt-location stages or `"complete"`

Actions:
1. Get ordered alternate locations via `_get_ordered_alt_locations()`
2. If no alternate locations exist: call `_set_stage("complete")` and exit
3. If alternate locations exist: execute alt-location sub-workflow (Section 5.5) for each location in order

**R5.2.6** The point-lens workflow must execute all applicable stages in sequence on a single `fit()` call — it must not stop between stages unless an exception occurs.

---

### 5.4 Binary-Lens Workflow

The binary-lens workflow reuses the point-lens stages through `"primary_renorm_done"`, then executes binary-specific stages. Space datasets are held in reserve and not used until after `"post_binary_renorm_done"`.

#### Requirements

**R5.3.1** `fit_binary_lens()` must first execute all point-lens stages through `"primary_renorm_done"` using **primary location datasets only**, regardless of whether space datasets are present.

**R5.3.1a** fit_point_lens() must accept an optional stop_at parameter of type Optional[str], defaulting to None. When stop_at is provided, fit_point_lens() must halt after reaching that stage and not proceed further. fit_binary_lens() must call fit_point_lens(stop_at="primary_renorm_done").

Add the corresponding test in test_binary_lens_workflow.py:

python

Copy code
def test_fit_point_lens_stops_at_requested_stage():
    """R5.3.1a: fit_point_lens() halts at stop_at stage when provided."""


**R5.3.2** Space datasets must not be included in any fit until after `"post_binary_renorm_done"` is reached.

**R5.3.3** `"primary_renorm_done"` → `"anomaly_search_done"`

Actions:
1. Compute residuals from best point-lens model using primary location datasets
2. Run AnomalyFinder grid search on residuals
3. Store `best_af_grid_point`
4. Compute and store `anomaly_lc_params` from `best_af_grid_point` and the result of `_select_preferred_point_lens()` (R3.4.6)
5. Call `_set_stage("anomaly_search_done")`

Skip condition: if `workflow_stage` is already at or past `"anomaly_search_done"`, skip.

**R5.3.4** `"anomaly_search_done"` → `"binary_fit_done"`

Actions:
1. Fit binary lens model seeded from `anomaly_lc_params` and best point-lens model
2. Merge duplicate binary solutions via `_merge_duplicate_solutions()` (R3.3.1)
3. Log original and merged solution counts (R3.3.5)
4. Assign `solution_index` values to surviving solutions ordered by chi2 (R3.3.6)
5. Call `_set_stage("binary_fit_done")`

Skip condition: if `workflow_stage` is already at or past `"binary_fit_done"`, skip.

**R5.3.5** `"binary_fit_done"` → `"post_binary_renorm_done"`

Actions:
1. Call `_needs_renormalization()` with best binary model (R3.5.1)
2. If `True`:
   a. Select reference model via `_select_reference_model()` (returns best binary)
   b. Run outlier removal and errfac calculation on all primary location datasets
   c. Apply errfacs, recreate datasets
   d. Mark all fits `needs_refit=True`
   e. Refit all `needs_refit=True` fits using current `params` as seeds
3. Call `_set_stage("post_binary_renorm_done")`

Skip condition: if `workflow_stage` is already at or past `"post_binary_renorm_done"`, skip.

**R5.3.6** `"post_binary_renorm_done"` → alt-location stages or `"complete"`

Actions:
1. Get ordered alternate locations via `_get_ordered_alt_locations()`
2. If no alternate locations exist: call `_set_stage("complete")` and exit
3. If alternate locations exist: execute alt-location sub-workflow (Section 5.5) for each location in order
4. Note: the coarse grid in the alt-location sub-workflow must be seeded from `_select_reference_model()` which returns the best binary model (R3.4.1)

---

### 5.5 Alt-Location Sub-Workflow

This sub-workflow is called once per alternate location, in the order defined by `alt_location_order`. It applies to both point-lens and binary workflows.

#### Requirements

**R5.4.1** For each alternate location `loc` in order, execute:

**Step 1:** current stage → `("alt_loc_added", [..., loc])`

Actions:
1. Fit static PSPL with all datasets (primary + all processed alternate locations so far + `loc`)
2. Run coarse parallax grid with all datasets, seeded from _find_related_fit(). If same-n_loc fits exist, _find_related_fit() returns solution_index=0 of the best existing n_loc configuration. If no multi-location fits exist yet, _find_related_fit() falls back to _select_preferred_point_lens().
3. Optimize all minima found in grid
4. Merge duplicate solutions via `_merge_duplicate_solutions()`
5. Assign `solution_index` values to surviving solutions ordered by chi2
6. Call `_set_stage(("alt_loc_added", [..., loc]))`

**Step 2:** `("alt_loc_added", [..., loc])` → `("alt_loc_renorm_done", [..., loc])`

Actions:
1. Select reference model via `_select_reference_model()`
2. Run outlier removal and errfac calculation on datasets from `loc` only (not yet in `renorm_factors`)
3. Apply errfacs, recreate datasets
4. Mark all multi-location parallax fits that include datasets from loc as needs_refit=True
5. Refit all `needs_refit=True` fits using current `params` as seeds
6. If `parallax_grid=True`: run final parallax grid
7. Call `_set_stage(("alt_loc_renorm_done", [..., loc]))`

**R5.4.2** After all alternate locations have been processed, call `_set_stage("complete")`.

**R5.4.3** The coarse parallax grid in Step 1 must always run regardless of `parallax_grid` setting — it is required to find initial multi-location solutions. The final parallax grid in Step 2 runs only if `parallax_grid=True`.

---

### 5.6 `fit_point_lens()` and `fit_binary_lens()` Interface

```python
class MMEXOFASTFitter:

    def fit(self) -> None:
        """
        Main entry point. Routes to fit_point_lens() or fit_binary_lens()
        based on fit_type. Handles pre- and post-workflow steps.
        """
        ...

    def fit_point_lens(self) -> None:
        """
        Execute all point-lens workflow stages from current workflow_stage
        through 'complete'. Skips already-completed stages.
        """
        ...

    def fit_binary_lens(self) -> None:
        """
        Execute all binary-lens workflow stages from current workflow_stage
        through 'complete'. Calls fit_point_lens() for shared stages,
        then executes binary-specific stages.
        """
        ...

    def _run_alt_location_subworkflow(self, location: str) -> None:
        """
        Execute the two-step alt-location sub-workflow for a single location.
        """
        ...
```

---

### 5.7 Tests

```python
# test_fit_routing.py

def test_fit_raises_without_fit_type():
    """R5.1.6: fit() raises ValueError when fit_type is None."""

def test_fit_calls_mark_stale_fits_first():
    """R5.1.1: _mark_stale_fits() called before any workflow steps."""

def test_fit_detects_new_locations():
    """R5.1.1: New locations detected and dataset addition workflow triggered."""

def test_fit_routes_to_point_lens():
    """R5.1.2: fit() calls fit_point_lens() for fit_type='point lens'."""

def test_fit_routes_to_binary_lens():
    """R5.1.3: fit() calls fit_binary_lens() for fit_type='binary lens'."""

def test_fit_checks_renorm_completeness_at_end():
    """R5.1.5: _check_renorm_completeness() called after workflow completes."""

def test_fit_outputs_latex_table_at_end():
    """R5.1.5: _output_latex_table() called after workflow completes."""


# test_point_lens_workflow.py

def test_start_to_event_search_done():
    """R5.2.1: EF grid search runs and stage advances to event_search_done."""

def test_event_search_done_skipped_if_already_complete():
    """R5.2.1: EF grid not re-run if stage already past event_search_done."""

def test_static_fit_done_fits_pspl():
    """R5.2.2: Static PSPL fit runs and stage advances to static_fit_done."""

def test_static_fit_done_fits_fspl_if_finite_source():
    """R5.2.2: Static FSPL fit runs when finite_source=True."""

def test_static_fit_done_skips_fspl_if_not_finite_source():
    """R5.2.2: Static FSPL fit skipped when finite_source=False."""

def test_pl_fit_done_fits_parallax_branches():
    """R5.2.3: PAR_u0+ and PAR_u0- fits run and stage advances to pl_fit_done."""

def test_pl_fit_done_seeds_from_static_model():
    """R5.2.3: Parallax fits seeded from _select_preferred_static_point_lens_model()."""

def test_primary_renorm_done_renormalizes_when_enabled():
    """R5.2.4: Renormalization runs when renormalize_errors=True."""

def test_primary_renorm_done_skips_renorm_when_disabled():
    """R5.2.4: Renormalization skipped when renormalize_errors=False."""

def test_primary_renorm_done_runs_parallax_grid_if_enabled():
    """R5.2.4: Parallax grid runs after renorm when parallax_grid=True."""

def test_primary_renorm_done_skips_grid_if_disabled():
    """R5.2.4: Parallax grid skipped when parallax_grid=False."""

def test_point_lens_reaches_complete_with_no_alt_locations():
    """R5.2.5: Stage advances to complete when no alternate locations exist."""

def test_point_lens_executes_all_stages_in_single_fit_call():
    """R5.2.6: All stages execute in one fit() call (one-touch)."""

def test_point_lens_resumes_from_intermediate_stage():
    """R5.2.4: Workflow resumes from current stage, skips completed stages."""


# test_binary_lens_workflow.py

def test_binary_uses_primary_location_only_through_renorm():
    """R5.3.1, R5.3.2: Space datasets not used until post_binary_renorm_done."""

def test_anomaly_search_done_runs_af_grid():
    """R5.3.3: AnomalyFinder grid runs and stage advances to anomaly_search_done."""

def test_anomaly_search_done_stores_af_grid_point():
    """R5.3.3: best_af_grid_point stored after anomaly search."""

def test_binary_fit_done_merges_duplicate_solutions():
    """R5.3.4: Duplicate binary solutions merged."""

def test_binary_fit_done_logs_merge_results():
    """R5.3.4: Original and merged solution counts logged."""

def test_binary_fit_done_assigns_solution_indices():
    """R5.3.4: solution_index assigned by chi2 after merging."""

def test_post_binary_renorm_runs_when_needed():
    """R5.3.5: Re-renormalization runs when _needs_renormalization() returns True."""

def test_post_binary_renorm_skips_when_not_needed():
    """R5.3.5: Re-renormalization skipped when _needs_renormalization() returns False."""

def test_binary_reaches_complete_with_no_alt_locations():
    """R5.3.6: Stage advances to complete when no alternate locations."""

def test_binary_alt_location_uses_binary_reference_model():
    """R5.3.6: Coarse grid in alt-location sub-workflow seeded from best binary model."""


# test_alt_location_subworkflow.py

def test_alt_loc_step1_fits_static_pspl_with_all_data():
    """R5.4.1 Step 1: Static PSPL fit uses all datasets including new location."""

def test_alt_loc_step1_runs_coarse_grid():
    """R5.4.1 Step 1: Coarse parallax grid always runs."""

def test_alt_loc_step1_merges_duplicate_solutions():
    """R5.4.1 Step 1: Duplicate solutions merged."""

def test_alt_loc_step1_assigns_solution_indices():
    """R5.4.1 Step 1: solution_index assigned by chi2."""

def test_alt_loc_step2_renormalizes_new_location_only():
    """R5.4.1 Step 2: Only new location datasets renormalized."""

def test_alt_loc_step2_refits_stale_fits():
    """R5.4.1 Step 2: Multi-location fits marked needs_refit and refit."""

def test_alt_loc_step2_runs_final_grid_if_enabled():
    """R5.4.3: Final parallax grid runs in Step 2 when parallax_grid=True."""

def test_alt_loc_step2_skips_final_grid_if_disabled():
    """R5.4.3: Final parallax grid skipped in Step 2 when parallax_grid=False."""

def test_alt_loc_coarse_grid_always_runs():
    """R5.4.3: Coarse grid runs regardless of parallax_grid setting."""

def test_multiple_alt_locations_processed_in_order():
    """R5.4.1, R5.4.2: Multiple alternate locations processed in alt_location_order."""

def test_complete_stage_set_after_all_alt_locations():
    """R5.4.2: Stage advances to complete after all alternate locations processed."""

def test_alt_loc_step1_coarse_grid_seeded_from_find_related_fit():
    """R5.4.1 Step 1: Coarse grid seeded from _find_related_fit(), 
    not _select_reference_model()."""

def test_alt_loc_step1_coarse_grid_seeded_from_point_lens_when_first_alt_loc():
    """R5.4.1 Step 1, R3.8.3: When no multi-location fits exist, coarse grid
    seeded from _select_preferred_point_lens()."""

def test_alt_loc_step1_coarse_grid_seeded_from_prev_solution_when_subsequent_alt_loc():
    """R5.4.1 Step 1, R3.8.3: When multi-location fits exist, coarse grid
    seeded from solution_index=0 of best existing n_loc."""
    
# test_workflow_integration.py

def test_ground_only_point_lens_full_workflow():
    """Integration: Complete ground-only point-lens workflow from start to complete."""

def test_ground_plus_space_point_lens_one_touch():
    """Integration: One-touch ground+space point-lens workflow."""

def test_ground_only_incremental_add_followup():
    """Integration: Incremental workflow with ground followup added mid-workflow."""

def test_ground_plus_space_incremental():
    """Integration: Incremental workflow adding space data after ground complete."""

def test_binary_lens_ground_only():
    """Integration: Complete ground-only binary lens workflow."""

def test_binary_lens_with_space_data():
    """Integration: Binary lens workflow with space data added after binary fit."""

def test_restart_from_intermediate_stage_point_lens():
    """Integration: Workflow correctly resumes from saved intermediate stage."""

def test_restart_from_intermediate_stage_binary_lens():
    """Integration: Binary workflow correctly resumes from saved intermediate stage."""

def test_fit_type_change_pl_to_binary_on_restart():
    """Integration: Restarting with fit_type changed from PL to binary preserves PL fits."""
```
## 5.8 Future Considerations

**R5.5.1** A future alternate binary workflow may be required in which space data are incorporated before the anomaly search. This would be triggered when the ground-only AnomalyFinder finds no significant anomaly. The current spec does not implement this workflow, but the architecture must not preclude it.

To ensure compatibility with this future workflow, the following constraints apply:

- The alt-location sub-workflow (`_run_alt_location_subworkflow()`) must remain callable from any stage, not only after `"post_binary_renorm_done"`
- The stage sequence must be designed so that new stages can be inserted between `"primary_renorm_done"` and `"anomaly_search_done"` without breaking existing behavior
- The AnomalyFinder result (`best_af_grid_point`) must not be assumed to have been computed from ground-only data — future implementations may compute it from multi-location data

**Note to developer:** When implementing the stage sequencing logic in `_is_valid_stage_progression()` and `_set_stage()`, do not hard-code assumptions about which datasets are active at the time of the anomaly search. The dataset selection at each stage should be passed explicitly rather than inferred from the stage name.

---
## Section 6: Output System — Specification

---

### 6.1 Background

The output system handles logging, restart state saving, LaTeX table generation, plot saving, and grid result saving. Several changes are needed: the unrenormalized dataset warning must be integrated at the right points, the LaTeX table output must handle incomplete workflows gracefully, and `initialize_exozippy()` must warn before returning results from a partially-complete workflow.

---

### 6.2 Logging

#### Requirements

**R6.1.1** All workflow stage transitions must be logged via `_set_stage()` (already specified in R2.13). The log entry must clearly state the old stage and the new stage.

**R6.1.2** At the start of every `fit()` call, the current `workflow_stage` must be logged before any steps execute.

**R6.1.3** When fits are marked `needs_refit=True` by `_mark_stale_fits()`, the log must state:
- Which fits were marked (by model label)
- Why they were marked (renorm factor changed, or same-location dataset added)
- What stage was reset to

**R6.1.4** When solution merging occurs, the log must state (R3.3.5):
- Original number of solutions
- Number of unique solutions after merging
- For each merged group: parameters of all discarded solutions

**R6.1.5** When `_needs_renormalization()` is called, the log must state the errfac for each dataset regardless of whether re-renormalization is triggered (R3.5.5).

**R6.1.6** The unrenormalized dataset warning must be written to the log at the end of `fit()` when renormalization has been partially applied (R3.7.2). The warning must list the specific unrenormalized dataset labels.

**R6.1.7** All existing logging behavior must be preserved.

---

### 6.3 Restart State

#### Requirements

**R6.2.1** The restart state must include `workflow_stage` (R2.2). This is in addition to all existing state fields.

**R6.2.2** `_save_restart_state()` is called automatically by `_set_stage()` after every stage transition (R2.14). It must not be removed from other locations where it is currently called explicitly — these provide additional mid-stage checkpoints.

**R6.2.3** The restart state format must remain a pickle dict with `'config'` and `'state'` keys. The `'state'` dict gains `'workflow_stage'` as a new key. No other format changes are required.

**R6.2.4** Loading a restart file that does not contain `'workflow_stage'` (legacy restart file) must not raise an error. The missing stage must be handled by `_infer_stage_from_results()` as specified in R2.7.

---

### 6.4 LaTeX Table Output

#### Requirements

**R6.3.1** `_output_latex_table()` must be called at the end of every `fit()` call regardless of what stage was reached (R5.1.5).

**R6.3.2** If `all_fit_results` is empty, `_output_latex_table()` must return without writing any file. No error is raised.

**R6.3.3** Before writing the table, `_check_renorm_completeness()` must be called and any warning logged (R3.7.2).

**R6.3.4** The table must include all fits currently in `all_fit_results` regardless of `needs_refit` status or `is_complete` status, so the user has a full record of current state. Fits with `needs_refit=True` or `is_complete=False` must be visually distinguished in the table — for example, by appending `*` to the model label.

**R6.3.5** All existing table formatting and column ordering behavior must be preserved.

---

### 6.5 `initialize_exozippy()`

#### Requirements

**R6.4.1** Before returning results, `initialize_exozippy()` must call `_check_renorm_completeness()` and log any warning (R3.7.2).

**R6.4.2** `initialize_exozippy()` must not raise an error if renormalization is incomplete — it must log the warning and proceed. The decision to use potentially incomplete results is left to the user.

**R6.4.3** All existing behavior of `initialize_exozippy()` should preserve the return format and intent.

---

### 6.6 Plot Output

#### Requirements

**R6.5.1** All existing plot saving behavior must be preserved.

**R6.5.2** No new plots are required by this spec. Future plot additions (e.g., light curve with fitted model) are out of scope.

---

### 6.7 Tests

```python
# test_logging.py

def test_stage_transition_logged_with_old_and_new_stage():
    """R6.1.1: Log entry states both old and new stage on transition."""

def test_current_stage_logged_at_start_of_fit():
    """R6.1.2: Current workflow_stage logged before any steps execute."""

def test_stale_fits_log_which_fits_marked():
    """R6.1.3: Log states which fits were marked needs_refit=True."""

def test_stale_fits_log_reason():
    """R6.1.3: Log states reason for marking (renorm change or dataset addition)."""

def test_stale_fits_log_stage_reset():
    """R6.1.3: Log states what stage was reset to."""

def test_merge_log_states_original_count():
    """R6.1.4: Log states original number of solutions before merging."""

def test_merge_log_states_final_count():
    """R6.1.4: Log states number of unique solutions after merging."""

def test_merge_log_states_discarded_params():
    """R6.1.4: Log states parameters of each discarded solution."""

def test_needs_renorm_logs_errfac_for_all_datasets():
    """R6.1.5: errfac logged for all datasets when _needs_renormalization() called."""

def test_unrenorm_warning_written_to_log():
    """R6.1.6: Warning written to log when renormalization partially applied."""

def test_unrenorm_warning_lists_dataset_labels():
    """R6.1.6: Warning lists specific unrenormalized dataset labels."""


# test_restart_state.py

def test_workflow_stage_included_in_state():
    """R6.2.1: workflow_stage present in _get_state() output."""

def test_workflow_stage_restored_from_state():
    """R6.2.1: workflow_stage correctly restored by _restore_state()."""

def test_restart_state_format_unchanged():
    """R6.2.3: Restart file format remains pickle with config and state keys."""

def test_legacy_restart_file_without_stage_loads_cleanly():
    """R6.2.4: Loading restart file without workflow_stage does not raise."""

def test_legacy_restart_file_triggers_inference():
    """R6.2.4: Missing workflow_stage triggers _infer_stage_from_results()."""

def test_restart_state_saved_on_stage_transition():
    """R6.2.2: _save_restart_state() called after every _set_stage()."""


# test_latex_output.py

def test_output_called_at_end_of_fit():
    """R6.3.1: _output_latex_table() called at end of every fit() call."""

def test_output_skipped_when_no_fits():
    """R6.3.2: No file written when all_fit_results is empty."""

def test_output_checks_renorm_completeness():
    """R6.3.3: _check_renorm_completeness() called before writing table."""

def test_needs_refit_fits_marked_in_table():
    """R6.3.4: Fits with needs_refit=True visually distinguished in table."""

def test_incomplete_fits_marked_in_table():
    """R6.3.4: Fits with is_complete=False visually distinguished in table."""

def test_all_fits_included_regardless_of_status():
    """R6.3.4: All fits in all_fit_results included in table output."""

def test_table_formatting_preserved():
    """R6.3.5: Existing column ordering and formatting unchanged."""


# test_initialize_exozippy.py

def test_initialize_exozippy_checks_renorm_completeness():
    """R6.4.1: _check_renorm_completeness() called before returning results."""

def test_initialize_exozippy_logs_warning_if_incomplete():
    """R6.4.1: Warning logged when renormalization incomplete."""

def test_initialize_exozippy_does_not_raise_if_incomplete():
    """R6.4.2: No error raised when renormalization incomplete."""

def test_initialize_exozippy_returns_results_when_incomplete():
    """R6.4.2: Results returned despite incomplete renormalization."""

def test_initialize_exozippy_existing_behavior_preserved():
    """R6.4.3: Existing return format and behavior unchanged."""
```

---
