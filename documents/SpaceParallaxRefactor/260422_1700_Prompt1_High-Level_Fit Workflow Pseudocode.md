# Conversation
- conversationId: 690fe7bb-9a96-43e1-ad2b-64913bd193e1
- endpoint: bedrock
- title: High-Level Fit Workflow Pseudocode
- exportAt: 08:28:08 GMT-0400 (Eastern Daylight Time)

## Options
- presetId: null
- modelLabel: Claude 4.6 Sonnet
- resendFiles: true
- iconURL: https://static.sandbox.ai.huit.harvard.edu/img/anthropic.png
- greeting: Hello! I'm Claude 4.6 Sonnet, running in the Harvard AI Sandbox. I have a 200k token context window (~150k words) and my knowledge cutoff is August 2025. How can I help you today?
- spec: claude-4.6-sonnet
- maxContextTokens: 900000
- model: us.anthropic.claude-sonnet-4-6
- promptCache: false
- endpoint: bedrock
- endpointType: undefined
- title: High-Level Fit Workflow Pseudocode

## History
**User**
# PHASE 1: SKETCH THE HIGHEST-LEVEL FLOW

## LLM INTERACTION GUIDELINES

When working through this phase with the user:

- Ask one question at a time
- Label possible answers A, B, C (or similar) for clarity
- Never assume details; ask for more information instead of guessing
- If a specification is ambiguous, ask for clarification before proceeding
- If you're about to make a significant decision, ask the user first
- Explicitly state what you don't understand or need more context for
- Wait for user confirmation before considering the phase complete
- Flag any contradictions or unclear requirements immediately

---

## CODE CONVENTIONS TO FOLLOW

When creating pseudocode and sketches, reference these conventions from the 
existing codebase:

**Docstrings:**
- Use NumPy format with Parameters, Returns, Raises sections
- Use `----------` underlines for section headers

**Naming:**
- Private methods: Prefix with `_`
- Config attributes: Listed in `CONFIG_KEYS` class variable

**State Management:**
- Use `_get_state()` and `_restore_state()` for pickling

**Output:**
- Use `self.output.log()` for messages
- Use `self.output.save_plot()` for figures

**Grid Searches:**
- Inherit from `BaseRectGridSearch`
- Implement `_setup_grid()` and `_fit_grid_point()`

**Floating Point:**
- All floating point grid coordinates should be rounded to 10 decimal places

---

## YOUR TASK

You are helping sketch the highest-level flow of a refactored workflow.

Starting from the public API entry point (or main workflow trigger), map out how the 
system should work at the highest level of abstraction.

**For the main entry method:**

1. Write pseudocode showing the major steps it must perform
2. Identify each step's responsibility (what does it accomplish?)
3. Sketch data flow between steps (what's passed from one to the next?)
4. Note where complexity lives (which steps will likely need sub-methods?)
5. List any assumptions you're making about how this should work

**Format your pseudocode like this:**

```
def main_method_name(input_data):
    """
    High-level description of what the workflow accomplishes.
    """
    # Step 1: Validate and normalize input
    #   - Takes: raw input_data
    #   - Produces: clean, validated data structure
    #   - Risk: edge cases in validation
    
    # Step 2: Do core transformation [COMPLEX - will likely need sub-methods]
    #   - Takes: validated data
    #   - Produces: transformed output
    #   - Risk: business logic is intricate, multiple scenarios
    
    # Step 3: Apply business rules
    #   - Takes: transformed data
    #   - Produces: final result
    #   - Risk: rules may conflict, need careful ordering
    
    # Step 4: Return result
    #   - Takes: final result
    #   - Produces: API response
```

Mark steps as [SIMPLE] or [COMPLEX] based on whether they'll need sub-methods.

---

## CONTEXT PROVIDED

### Specifications:
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


### Existing Code:
"""
High-level functions for fitting microlensing events.
"""
"""
mmexofast_fitter_arch.py

Architectural sketch for MMEXOFASTFitter with:
- Structured model keys
- Centralized fit result registry (mmexo.AllFitResults)
- mmexo.FitRecord for partial/user-supplied vs full results
"""

from typing import Dict, Any, Optional, Iterable
import pickle
import inspect

import pandas as pd
from scipy.special import erfcinv
from scipy.optimize import minimize_scalar
import numpy as np
import os.path

import MulensModel

import exozippy.mmexofast as mmexo


# ============================================================================
# MMEXOFASTFitter
# ============================================================================
def fit(files=None, fit_type=None, **kwargs):
    """
    Fit a microlensing light curve using MMEXOFAST.

    Parameters
    ----------
    files : str or list, optional
        Data file(s) to fit
    fit_type : str, optional
        Type of fit ('point lens', 'binary lens')
    **kwargs : dict
        Additional arguments passed to MMEXOFASTFitter

    Returns
    -------
    MMEXOFASTFitter
        Fitted fitter object
    """
    fitter = MMEXOFASTFitter(files=files, fit_type=fit_type, **kwargs)
    fitter.fit()
    return fitter


class MMEXOFASTFitter:
    """
    Orchestrates workflows (PSPL, parallax, binary, etc.) and uses:
    - mmexo.ModelKey to identify models
    - mmexo.AllFitResults to store/reuse results
    """

    CONFIG_KEYS = [
        'files', 'fit_type', 'finite_source', 'coords', 'mag_methods',
        'limb_darkening_coeffs_u', 'limb_darkening_coeffs_gamma',
        'renormalize_errors', 'parallax_grid', 'verbose', 'fix_blend_flux',
        'fix_source_flux', 'primary_location', 'primary_dataset'
    ]

    # Parallax grid search parameters
    PARALLAX_GRID_PARAMS_COARSE = {
        'pi_E_E_min': -1.0,
        'pi_E_E_max': 1.0,
        'pi_E_E_step': 0.15,
        'pi_E_N_min': -1.5,
        'pi_E_N_max': 1.5,
        'pi_E_N_step': 0.3
    }

    PARALLAX_GRID_PARAMS_FINE = {
        'pi_E_E_min': -0.7,
        'pi_E_E_max': 0.7,
        'pi_E_E_step': 0.025,
        'pi_E_N_min': -1.0,
        'pi_E_N_max': 1.0,
        'pi_E_N_step': 0.05
    }

    def __init__(
            self,
            files=None,
            datasets=None,
            fit_type=None,
            finite_source=False,
            coords=None,
            mag_methods=None,
            limb_darkening_coeffs_u=None,
            limb_darkening_coeffs_gamma=None,
            fix_blend_flux=None,
            fix_source_flux=None,
            renormalize_errors=False,
            parallax_grid=False,
            verbose=False,
            initial_results=None,
            output_config=None,
            restart_file=None,
    ):
        # Validate mutually exclusive parameters
        if files is not None and datasets is not None:
            raise ValueError("Cannot specify both 'files' and 'datasets'. Provide only one.")

        # Ensure output_config exists
        if output_config is None:
            output_config = mmexo.OutputConfig()  # Uses default values

        # Output
        self.verbose = verbose
        self.output = mmexo.OutputManager(output_config, verbose=self.verbose) if output_config is not None else None

        # Load restart data
        saved_config, saved_state = self._load_restart_data(restart_file)
        config = self._merge_config(saved_config, locals())
        self._set_config_attributes(config)

        # Restore state
        self._restore_state(saved_state)

        # Create or use datasets
        if files:
            self.datasets = self._create_mulensdata_objects(
                files, saved_datasets=saved_state.get('datasets')
            )
        elif datasets:
            self.datasets = datasets
            self._validate_dataset_labels()  # NEW: Validate user-provided datasets
            if saved_state.get('datasets'):
                self._merge_with_saved_datasets(saved_state['datasets'])
        elif saved_state.get('datasets'):
            # Using only restart file datasets
            self.datasets = saved_state['datasets']
        else:
            raise ValueError("Must provide files, datasets, or restart_file")

        # Verify dataset labels are unique
        self._check_dataset_labels_unique()

        # Recalculate n_loc based on current datasets
        old_n_loc = saved_state.get('n_loc')
        self.n_loc = self._count_loc()
        self._location_groups = None

        # If datasets were updated, old fit results need to be refit
        self._datasets_changed = False
        if (files or datasets) and saved_state.get('all_fit_results'):
            self._datasets_changed = True

            # If n_loc changed, also remove parallax fits (wrong branches)
            #if old_n_loc is not None and old_n_loc != self.n_loc:
            #    self._remove_parallax_fits()

        # Map flux fixing options using label mapping
        self.fix_blend_flux_map = self._map_label_dict_to_datasets(self.fix_blend_flux)
        self.fix_source_flux_map = self._map_label_dict_to_datasets(self.fix_source_flux)

        self.residuals = None

        if self.parallax_grid:
            if not (self.output.config.save_plots or self.output.config.save_grid_results):
                raise ValueError(
                    "parallax_grid is enabled but neither save_plots nor save_grid_results "
                    "is set in output config. At least one must be enabled to use parallax_grid."
                )

        # Load initial results if provided
        if initial_results is not None:
            self._load_initial_results(initial_results)

    # ---------------------------------------------------------------------
    # restart helpers:
    # ---------------------------------------------------------------------
    def _get_config(self) -> dict:
        """
        Automatically extract config from attributes.

        Returns
        -------
        dict
            Configuration dictionary with all CONFIG_KEYS
        """
        return {key: getattr(self, key, None) for key in self.CONFIG_KEYS}

    def _merge_config(self, saved_config, provided_params):
        """
        Merge saved config with provided params (provided wins).

        Parameters
        ----------
        saved_config : dict
            Configuration from restart file
        provided_params : dict
            Parameters provided to __init__

        Returns
        -------
        dict
            Merged configuration
        """
        merged = {}
        for key in self.CONFIG_KEYS:
            if key in provided_params and provided_params[key] is not None:
                merged[key] = provided_params[key]
            elif key in saved_config:
                merged[key] = saved_config[key]
            else:
                merged[key] = None
        return merged

    def _set_config_attributes(self, config):
        """
        Set all config attributes from config dict.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        for key in self.CONFIG_KEYS:
            setattr(self, key, config[key])

    def _get_fitter_kwargs(self) -> dict:
        """
        Bundle fitter options for passing to SFitFitter.

        Returns
        -------
        dict
            Fitter configuration options
        """
        return {
            'coords': self.coords,
            'mag_methods': self.mag_methods,
            'limb_darkening_coeffs_u': self.limb_darkening_coeffs_u,
            'limb_darkening_coeffs_gamma': self.limb_darkening_coeffs_gamma,
            'fix_source_flux': self.fix_source_flux_map,
            'fix_blend_flux': self.fix_blend_flux_map
        }

    def _get_state(self) -> dict:
        """
        Get all computed state (fit results).

        Returns
        -------
        dict
            State dictionary for pickling
        """
        return {
            'all_fit_results': self.all_fit_results,
            'best_ef_grid_point': self.best_ef_grid_point,
            'best_af_grid_point': self.best_af_grid_point,
            'anomaly_lc_params': self.anomaly_lc_params,
            'n_loc': self.n_loc,
            'datasets': self.datasets,
            'renorm_factors': self.renorm_factors,
        }

    def _load_restart_data(self, restart_file):
        """
        Load config and state from restart file.

        Parameters
        ----------
        restart_file : str or None
            Path to restart pickle file

        Returns
        -------
        tuple
            (config_dict, state_dict)
        """
        if restart_file is None:
            return {}, {}

        self._log(f"Loading restart data from: {restart_file}")

        with open(restart_file, 'rb') as f:
            data = pickle.load(f)

        config = data.get('config', {})
        state = data.get('state', {})

        # Log what was loaded
        n_datasets = len(state.get('datasets', []))
        n_fits = len(state.get('all_fit_results', mmexo.AllFitResults()))
        n_renorm = len(state.get('renorm_factors', {}))

        self._log(f"  Loaded {n_datasets} datasets")
        self._log(f"  Loaded {n_fits} fit results")
        self._log(f"  Loaded {n_renorm} renormalization factors")

        if n_renorm > 0:
            self._log(f"  Renormalized datasets: {list(state.get('renorm_factors', {}).keys())}")

        return config, state

    def _restore_state(self, saved_state):
        """
        Restore computed state from saved data.

        Parameters
        ----------
        saved_state : dict
            State dictionary from restart file
        """
        self.all_fit_results = saved_state.get('all_fit_results', mmexo.AllFitResults())
        self.best_ef_grid_point = saved_state.get('best_ef_grid_point')
        self.best_af_grid_point = saved_state.get('best_af_grid_point')
        self.anomaly_lc_params = saved_state.get('anomaly_lc_params')
        self.renorm_factors = saved_state.get('renorm_factors', {})

    def _load_initial_results(self, initial_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Load user-supplied initial results into mmexo.AllFitResults.

        Expected format for each entry:
        {
            "params": {...},              # required
            "sigmas": {...},              # optional
            "renorm_factors": {...},      # optional
            "fixed": bool,                # optional
        }

        Parameters
        ----------
        initial_results : dict
            Dictionary mapping model labels to result dictionaries
        """
        for label, payload in initial_results.items():
            key = mmexo.fit_types.label_to_model_key(label)
            record = mmexo.FitRecord(
                model_key=key,
                params=payload["params"],
                sigmas=payload.get("sigmas"),
                renorm_factors=payload.get("renorm_factors"),
                full_result=None,
                fixed=payload.get("fixed", False),
                is_complete=False,
            )
            self.all_fit_results.set(record)

   # def _remove_parallax_fits(self):
   #     """
   #     Remove parallax fits from all_fit_results.
   #
   #     Called when n_loc changes, since parallax branches depend on n_loc.
   #     Keeps static point lens models (PSPL/FSPL) which are n_loc-independent.
   #     """
   #     keys_to_remove = []
   #
   #     for key in self.all_fit_results:
   #         # Remove if parallax branch is not NONE
   #         if key.parallax_branch != mmexo.ParallaxBranch.NONE:
   #             keys_to_remove.append(key)
   #
   #     for key in keys_to_remove:
   #         del self.all_fit_results[key]

    def _infer_workflow_state(self):
        """
        Infer the current workflow state from available data.

        Returns
        -------
        dict
            Dictionary containing workflow state information:
            - n_loc: Number of observing locations
            - primary_location: Inferred primary location name
            - renorm_by_location: Dict mapping location to renormalized/not_renormalized dataset labels
            - has_static_fits: Whether static point lens fits exist
            - has_parallax_fits: Whether parallax fits exist
            - locations_in_static_fits: Which locations were used in static fits
            - locations_in_parallax_fits: List of locations used in parallax fits
            - complete_fit_labels: List of completed fit model labels
            - incomplete_fit_labels: List of incomplete fit model labels
        """
        workflow_state = {}

        # Basic info
        workflow_state['n_loc'] = self.n_loc

        # Infer primary location (priority order)
        primary_location = None

        # 1. Check if already set from previous multi-location run
        if hasattr(self, '_primary_location') and self._primary_location is not None:
            primary_location = self._primary_location

        # 2. Check config parameter
        elif self.primary_location is not None:
            primary_location = self.primary_location

        # 3. Check primary_dataset config parameter
        elif self.primary_dataset is not None:
            # Find location for this dataset
            for loc, datasets in self.location_groups.items():
                labels = [ds.plot_properties['label'] for ds in datasets]
                if self.primary_dataset in labels:
                    primary_location = loc
                    break

        # 4. Infer from renormalized datasets
        elif len(self.renorm_factors) > 0:
            # Find which location has renormalized datasets
            for loc, datasets in self.location_groups.items():
                labels = [ds.plot_properties['label'] for ds in datasets]
                if any(label in self.renorm_factors for label in labels):
                    primary_location = loc
                    break

        # 5. Infer from static fits
        elif self._select_best_static_pspl() is not None:
            best_static = self._select_preferred_static_point_lens_model()
            static_key = None
            for key, record in self.all_fit_results.items():
                if record == best_static:
                    static_key = key
                    break

            if static_key and static_key.locations_used:
                # Parse locations_used (could be 'ground', 'All', 'ground+Spitzer', etc.)
                if static_key.locations_used != 'All':
                    # Take first location mentioned
                    primary_location = static_key.locations_used.split('+')[0]

        # 6. Fall back to longest coverage
        if primary_location is None:
            primary_datasets = self._select_primary_location_by_coverage()
            for loc, datasets in self.location_groups.items():
                if set(datasets) == set(primary_datasets):
                    primary_location = loc
                    break

        workflow_state['primary_location'] = primary_location

        # Build renorm_by_location
        renorm_by_location = {}
        for loc, datasets in self.location_groups.items():
            labels = [ds.plot_properties['label'] for ds in datasets]
            renormalized = [label for label in labels if label in self.renorm_factors]
            not_renormalized = [label for label in labels if label not in self.renorm_factors]

            renorm_by_location[loc] = {
                'renormalized': renormalized,
                'not_renormalized': not_renormalized
            }

        workflow_state['renorm_by_location'] = renorm_by_location

        # Check for fits
        has_static = False
        has_parallax = False
        locations_in_static = set()
        locations_in_parallax = set()

        for key, record in self.all_fit_results.items():
            if key.parallax_branch == mmexo.ParallaxBranch.NONE:
                has_static = True
                if key.locations_used:
                    if key.locations_used == 'All':
                        locations_in_static.add('All')
                    else:
                        # Parse 'ground+Spitzer' format
                        for loc in key.locations_used.split('+'):
                            locations_in_static.add(loc)
            else:
                has_parallax = True
                if key.locations_used:
                    if key.locations_used == 'All':
                        locations_in_parallax.add('All')
                    else:
                        for loc in key.locations_used.split('+'):
                            locations_in_parallax.add(loc)

        workflow_state['has_static_fits'] = has_static
        workflow_state['has_parallax_fits'] = has_parallax

        # Convert to single string or None for static fits
        if len(locations_in_static) == 0:
            workflow_state['locations_in_static_fits'] = None
        elif 'All' in locations_in_static:
            workflow_state['locations_in_static_fits'] = 'All'
        else:
            workflow_state['locations_in_static_fits'] = '+'.join(sorted(locations_in_static))

        # Keep as list for parallax fits
        if 'All' in locations_in_parallax:
            workflow_state['locations_in_parallax_fits'] = ['All']
        else:
            workflow_state['locations_in_parallax_fits'] = sorted(list(locations_in_parallax))

        # Get complete and incomplete fit labels
        complete_fit_labels = []
        incomplete_fit_labels = []

        for key, record in self.all_fit_results.items():
            label = mmexo.fit_types.model_key_to_label(key)
            if record.is_complete:
                complete_fit_labels.append(label)
            else:
                incomplete_fit_labels.append(label)

        workflow_state['complete_fit_labels'] = complete_fit_labels
        workflow_state['incomplete_fit_labels'] = incomplete_fit_labels

        return workflow_state

    def _log_workflow_state(self, workflow_state=None):
        """
        Log the current workflow state for debugging.

        Parameters
        ----------
        workflow_state : dict or None, optional
            Workflow state dict from _infer_workflow_state().
            If None, calls _infer_workflow_state() to get current state.
        """
        if workflow_state is None:
            workflow_state = self._infer_workflow_state()

        self._log("\n" + "=" * 60)
        self._log("WORKFLOW STATE")
        self._log("=" * 60)

        self._log(f"Number of locations: {workflow_state['n_loc']}")
        self._log(f"Primary location: {workflow_state['primary_location']}")

        self._log("\nDatasets by location:")
        for loc, info in workflow_state['renorm_by_location'].items():
            n_renorm = len(info['renormalized'])
            n_not_renorm = len(info['not_renormalized'])
            self._log(f"  {loc}: {n_renorm} renormalized, {n_not_renorm} not renormalized")
            if n_renorm > 0:
                for label in info['renormalized']:
                    self._log(f"    ✓ {label}")
            if n_not_renorm > 0:
                for label in info['not_renormalized']:
                    self._log(f"    ✗ {label}")

        self._log("\nFit status:")
        self._log(f"  Has static fits: {workflow_state['has_static_fits']}")
        if workflow_state['has_static_fits']:
            self._log(f"    Locations: {workflow_state['locations_in_static_fits']}")

        self._log(f"  Has parallax fits: {workflow_state['has_parallax_fits']}")
        if workflow_state['has_parallax_fits']:
            self._log(f"    Locations: {', '.join(workflow_state['locations_in_parallax_fits'])}")

        n_complete = len(workflow_state['complete_fit_labels'])
        n_incomplete = len(workflow_state['incomplete_fit_labels'])
        self._log(f"\n  Complete fits: {n_complete}")
        if n_complete > 0:
            for label in workflow_state['complete_fit_labels']:
                self._log(f"    {label}")

        if n_incomplete > 0:
            self._log(f"  Incomplete fits: {n_incomplete}")
            for label in workflow_state['incomplete_fit_labels']:
                self._log(f"    {label}")

        self._log("=" * 60 + "\n")

    # ---------------------------------------------------------------------
    # Working with datasets:
    # ---------------------------------------------------------------------
    def _create_mulensdata_objects(self, files, saved_datasets=None):
        """
        Create MulensData objects, reusing saved datasets when labels match.

        Parameters
        ----------
        files : str or list
            File paths to load
        saved_datasets : list or None
            Previously saved datasets to reuse if labels match

        Returns
        -------
        list
            List of MulensData objects
        """
        if isinstance(files, str):
            files = [files]

        # Build mapping of saved datasets by label
        saved_by_label = {}
        if saved_datasets:
            for dataset in saved_datasets:
                label = dataset.plot_properties.get('label')
                if label:
                    saved_by_label[label] = dataset

        datasets = []

        for filename in files:
            # Extract label from filename (basename)
            label = os.path.basename(filename)

            # Check if we have a saved version with this label
            if label in saved_by_label:
                data = saved_by_label[label]
            else:
                # Load fresh from file
                if not os.path.exists(filename):
                    raise FileNotFoundError(f"Data file {filename} does not exist")

                kwargs = mmexo.observatories.get_kwargs(filename)
                data = MulensModel.MulensData(file_name=filename, **kwargs)
                # Label is already set by get_kwargs()

            datasets.append(data)

        return datasets

    def _validate_dataset_labels(self):
        """
        Validate that all user-provided datasets have labels set.

        For datasets with file_name but no label, sets label to basename.
        For datasets without file_name or label, raises an error.

        Raises
        ------
        ValueError
            If any dataset lacks both file_name and label
        """
        for i, dataset in enumerate(self.datasets):
            label = dataset.plot_properties.get('label')

            if not label:
                # Try to get from file_name
                if hasattr(dataset, 'file_name') and dataset.file_name:
                    label = os.path.basename(dataset.file_name)
                    dataset.plot_properties['label'] = label
                else:
                    raise ValueError(
                        f"Dataset at index {i} does not have a label set in "
                        f"plot_properties['label'] and was not created from a file. "
                        f"Please set dataset.plot_properties['label'] to a unique "
                        f"identifier before passing to MMEXOFASTFitter."
                    )

    def _map_label_dict_to_datasets(self, label_dict) -> dict:
        """
        Map a dict[label: value] to dict[dataset: value].

        Parameters
        ----------
        label_dict : dict or None
            Keys are dataset labels, values are bool or other values

        Returns
        -------
        dict
            Keys are MulensData objects, values from label_dict
        """
        if label_dict is None:
            # Default: False for all datasets
            return {dataset: False for dataset in self.datasets}

        result = {}
        for dataset in self.datasets:
            # Get label from dataset
            label = dataset.plot_properties.get('label')

            if label and label in label_dict:
                result[dataset] = label_dict[label]
            else:
                # Default if not specified
                result[dataset] = False

        return result

    def _check_dataset_labels_unique(self):
        """
        Verify that all dataset labels are unique.

        Raises
        ------
        ValueError
            If duplicate labels are found
        """
        labels = [ds.plot_properties.get('label') for ds in self.datasets]

        # Check for None labels
        if None in labels:
            raise ValueError(
                "Some datasets do not have labels set in plot_properties['label']. "
                "All datasets must have unique labels."
            )

        # Check for duplicates
        duplicates = [label for label in set(labels) if labels.count(label) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate dataset labels found: {duplicates}. "
                "All datasets must have unique labels in plot_properties['label']."
            )

    def _merge_with_saved_datasets(self, saved_datasets):
        """
        Replace current datasets with saved versions if labels match.

        This ensures renormalized datasets from restart_file are used instead
        of freshly loaded versions.

        Parameters
        ----------
        saved_datasets : list
            List of MulensData objects from restart file
        """
        # Build mapping: label -> saved dataset
        saved_by_label = {}
        for dataset in saved_datasets:
            label = dataset.plot_properties.get('label')
            if label:
                saved_by_label[label] = dataset

        # Replace matching datasets
        n_replaced = 0
        for i, dataset in enumerate(self.datasets):
            label = dataset.plot_properties.get('label')
            if label and label in saved_by_label:
                self.datasets[i] = saved_by_label[label]
                n_replaced += 1
                self._log(f"  Replaced dataset '{label}' with saved version")

        if n_replaced == 0:
            self._log("  No datasets replaced (no label matches)")
        else:
            self._log(f"  Replaced {n_replaced} dataset(s) with saved versions")

    # Location grouping
    def _count_loc(self):
        """
        Determine how many locations an event was observed from.

        Returns
        -------
        int
            Number of distinct observing locations
        """
        if len(self.datasets) == 1:
            return 1

        else:
            locs = []
            for dataset in self.datasets:
                if dataset.ephemerides_file not in locs:
                    locs.append(dataset.ephemerides_file)

            return len(locs)

    @property
    def location_groups(self):
        """
        Dictionary mapping location names to lists of datasets.

        Cached after first access. Keys are location names like 'ground',
        'Spitzer', or ephemerides file paths for unregistered observatories.

        Returns
        -------
        dict
            Keys are location names (str), values are lists of MulensData objects
        """
        if not hasattr(self, '_location_groups') or self._location_groups is None:
            self._location_groups = self._group_datasets_by_location()
        return self._location_groups

    def _group_datasets_by_location(self):
        """
        Group datasets by observing location.

        Returns
        -------
        dict
            Keys are location names ('ground', observatory names, or ephemerides paths).
            Values are lists of MulensData objects from that location.
        """
        groups = {}

        for dataset in self.datasets:
            ephem_file = getattr(dataset, 'ephemerides_file', None)

            if ephem_file is None:
                # Ground-based observation
                location = 'ground'
            elif ephem_file in mmexo.observatories.EPHEMERIDES_TO_OBSERVATORY:
                # Registered space observatory
                location = mmexo.observatories.EPHEMERIDES_TO_OBSERVATORY[ephem_file]
            else:
                # Unknown/custom ephemerides file
                location = ephem_file

            if location not in groups:
                groups[location] = []
            groups[location].append(dataset)

        return groups

    def _count_locations_used(self, locations_used):
        """
        Count number of locations represented in locations_used string.

        Parameters
        ----------
        locations_used : str or None
            Location identifier from FitKey

        Returns
        -------
        int
            Number of locations (higher is more complete)
        """
        if locations_used is None:
            return 1  # Single location when n_loc=1
        else:
            # Count locations in string like 'ground+Spitzer'
            return locations_used.count('+') + 1

    def _get_location_group_by_name(self, location_name):
        """
        Get datasets for a specific location by name.

        Parameters
        ----------
        location_name : str
            Location name ('ground', observatory name, or ephemerides path)

        Returns
        -------
        list
            Datasets from that location

        Raises
        ------
        ValueError
            If location name not found
        """
        groups = self._group_datasets_by_location()
        if location_name not in groups:
            available = list(groups.keys())
            raise ValueError(
                f"Location '{location_name}' not found. Available locations: {available}"
            )
        return groups[location_name]

    def _get_location_group_for_dataset(self, label):
        """
        Get the location group containing a specific dataset.

        Parameters
        ----------
        label : str
            Label of the dataset

        Returns
        -------
        list
            All datasets from the same location

        Raises
        ------
        ValueError
            If label not found
        """
        # Find the dataset with this label
        target_dataset = None
        for dataset in self.datasets:
            if dataset.plot_properties.get('label') == label:
                target_dataset = dataset
                break

        if target_dataset is None:
            raise ValueError(f"Dataset with label '{label}' not found")

        # Find which group it belongs to
        groups = self._group_datasets_by_location()
        for location, datasets in groups.items():
            if target_dataset in datasets:
                return datasets

        # Should never reach here
        raise ValueError(f"Dataset not found in any location group")

    def _get_location_time_coverage(self, location):
        """
        Calculate time coverage for a location.

        Parameters
        ----------
        location : str
            Location name

        Returns
        -------
        float
            Time span (max - min) for all datasets from this location
        """
        datasets = self.location_groups[location]
        all_times = np.concatenate([ds.time for ds in datasets])
        coverage = np.max(all_times) - np.min(all_times)
        return coverage

    def _select_primary_location_by_coverage(self):
        """
        Automatically select primary location based on time coverage.

        Returns
        -------
        list
            Datasets from the location with longest time coverage
        """
        groups = self._group_datasets_by_location()

        best_location = None
        max_coverage = 0.0

        for location in groups.keys():
            coverage = self._get_location_time_coverage(location)

            if coverage > max_coverage:
                max_coverage = coverage
                best_location = location

        return groups[best_location]

    def _get_location_for_datasets(self, datasets):
        """
        Determine which location(s) a set of datasets belongs to.

        Parameters
        ----------
        datasets : list
            List of MulensData objects

        Returns
        -------
        str or None
            Location name ('ground', 'Spitzer', etc.), 'All' if multiple
            locations when n_loc > 1, or None if single location (n_loc == 1)
        """
        dataset_set = set(datasets)

        # Check if using all datasets
        if dataset_set == set(self.datasets):
            return 'All' if self.n_loc > 1 else None

        # Find which location(s) match
        matching_locations = []
        for loc, loc_datasets in self.location_groups.items():
            if dataset_set == set(loc_datasets):
                matching_locations.append(loc)

        if len(matching_locations) == 1:
            return matching_locations[0]
        elif len(matching_locations) > 1:
            # Datasets span multiple locations
            return 'All' if self.n_loc > 1 else None
        else:
            # Partial subset - best guess
            return 'All' if self.n_loc > 1 else None

    # ---------------------------------------------------------------------
    # Public orchestration methods:
    # ---------------------------------------------------------------------
    def fit(self):
        """
        Perform the fit according to settings.
        """
        if self.fit_type is None:
            raise ValueError(
                'You must set the fit_type when initializing the ' +
                'MMEXOFASTFitter(): fit_type=("point lens", "binary lens")')

        if self.fit_type == 'point lens':
            # Use unified workflow for all cases (single or multi-location)
            self.fit_point_lens()

        elif self.fit_type == 'binary lens':
            # Binary lens workflow
            self.fit_binary_lens()

        self._output_latex_table()

    def fit_point_lens(self):
        """
        Run the unified point-lens workflow for single or multi-location data.

        Workflow:
        - Fit primary location static models (if needed)
        - Run initial parallax fitting (if needed)
        - Renormalize datasets (if needed)
        - Comprehensive parallax fitting (extract/optimize solutions or run detailed grid)
        """
        # ----------------------------------------------------------------
        # Infer current state and setup
        # ----------------------------------------------------------------
        state = self._infer_workflow_state()
        self._log_workflow_state(state)

        primary_loc = state['primary_location']
        self._primary_location = primary_loc  # Store for use by other methods
        all_datasets = self.datasets

        # ----------------------------------------------------------------
        # Nested workflow sections
        # ----------------------------------------------------------------
        def fit_primary_location_if_needed():
            """Fit static models for primary location if needed."""
            primary_needs_fitting = (
                    not state['has_static_fits'] or
                    (self.renormalize_errors and
                     len(state['renorm_by_location'][primary_loc]['not_renormalized']) > 0) or
                    (state['locations_in_static_fits'] and
                     primary_loc not in state['locations_in_static_fits'])
            )

            if not primary_needs_fitting:
                return

            self._log(f"\n{'=' * 60}")
            self._log("FITTING PRIMARY LOCATION")
            self._log(f"Primary location: {primary_loc}")
            self._log(f"{'=' * 60}")

            primary_datasets = self.location_groups[primary_loc]

            # Fit static models only (no renormalization yet)
            self._ensure_static_point_lens(primary_datasets)
            if self.finite_source:
                self._ensure_static_finite_point_lens(primary_datasets)

            self._save_restart_state()

        def run_initial_parallax_if_needed():
            """Run initial parallax grid or direct fitting based on n_loc."""
            # Determine if we need initial parallax fitting
            need_grid = (self.n_loc > 1 and
                         not (state['has_parallax_fits'] and self._parallax_fits_match_n_loc()))

            need_direct_fit = (self.n_loc == 1 and not state['has_parallax_fits'])

            if not (need_grid or need_direct_fit):
                return None  # Return None to signal no grids were created

            self._log(f"\n{'=' * 60}")
            self._log("INITIAL PARALLAX FITTING")
            self._log(f"{'=' * 60}")

            if need_grid:
                # Multi-location: run grid search
                self._log("\nRunning initial parallax grid with all datasets")

                initial_grids = self._run_piE_grid_search(
                    datasets=all_datasets,
                    grid_params=self.PARALLAX_GRID_PARAMS_COARSE,
                    skip_optimization=True,
                    save_results=True,
                    file_suffix='_initial'
                )

                # Optimize best solution
                self._log("\nOptimizing best parallax solution")
                best_solution = self._get_best_from_grids(initial_grids)
                parallax_fit = self._optimize_parallax_solution(best_solution[1], all_datasets)

                # Store temporarily in all_fit_results
                u_0 = best_solution[1]['u_0']
                if u_0 >= 0:
                    temp_branch = mmexo.ParallaxBranch.U0_PLUS
                else:
                    temp_branch = mmexo.ParallaxBranch.U0_MINUS

                temp_key = mmexo.FitKey(
                    lens_type=mmexo.LensType.POINT,
                    source_type=mmexo.SourceType.FINITE if self.finite_source else mmexo.SourceType.POINT,
                    parallax_branch=temp_branch,
                    lens_orb_motion=mmexo.LensOrbMotion.NONE,
                    locations_used=self._get_location_for_datasets(all_datasets)
                )

                temp_record = mmexo.FitRecord.from_full_result(
                    model_key=temp_key,
                    full_result=parallax_fit,
                    renorm_factors=self.renorm_factors,
                    fixed=False,
                )
                self.all_fit_results.set(temp_record)
                self._log(f"Stored temporary parallax fit: {temp_branch.value}, chi2={parallax_fit.chi2:.2f}")

                # Fit static model for all datasets (after parallax)
                self._log("\nFitting static model for all datasets")
                if self.finite_source:
                    self._ensure_static_finite_point_lens(all_datasets)
                else:
                    self._ensure_static_point_lens(all_datasets)

                self._save_restart_state()
                return initial_grids

            else:
                # Single location: fit branches directly
                self._log("\nFitting parallax branches directly (single location)")

                # Ensure static fit exists
                if self.finite_source:
                    self._ensure_static_finite_point_lens(all_datasets)
                else:
                    self._ensure_static_point_lens(all_datasets)

                # Fit parallax branches
                self._ensure_point_lens_parallax_models(all_datasets)

                self._save_restart_state()
                return None  # No grids created

        def renormalize_if_needed():
            """Renormalize unrennormalized datasets using best model."""
            has_unrennormalized = any(len(info['not_renormalized']) > 0
                                      for info in state['renorm_by_location'].values())

            if not (self.renormalize_errors and has_unrennormalized):
                return

            self._log(f"\n{'=' * 60}")
            self._log("RENORMALIZATION")
            self._log(f"{'=' * 60}")

            # Select best model (static or parallax) for reference
            reference_fit = self._select_preferred_point_lens()
            reference_model = reference_fit.full_result.fitter.get_model()

            self._log(
                f"Using reference model: {mmexo.fit_types.model_key_to_label(reference_fit.model_key)}, chi2={reference_fit.chi2():.2f}")

            # Renormalize all unrennormalized datasets
            error_factors = self._remove_outliers_and_calc_errfacs(
                reference_model,
                fit_datasets=self.datasets
            )
            self._apply_error_renormalization(error_factors)

            self._save_restart_state()

        def comprehensive_parallax_fitting(initial_grids):
            """Run comprehensive parallax fitting - final grid or optimize from initial."""
            self._log(f"\n{'=' * 60}")
            self._log("COMPREHENSIVE PARALLAX FITTING")
            self._log(f"{'=' * 60}")

            if self.parallax_grid:
                # Run final detailed grid
                self._log("\nRunning detailed parallax grid")
                final_grids = self._run_piE_grid_search(
                    datasets=all_datasets,
                    grid_params=self.PARALLAX_GRID_PARAMS_COARSE,
                    skip_optimization=False,
                    save_results=True,
                    file_suffix='_final',
                    refinement_params={'chi2_threshold': 9, 'min_step_size': 0.01, 'radius_steps': 3}
                )

                # Extract and optimize solutions (skip if single location with existing fits)
                if self.n_loc > 1 or not state['has_parallax_fits']:
                    self._extract_and_optimize_parallax_solutions(
                        final_grids[mmexo.ParallaxBranch.U0_PLUS],
                        final_grids[mmexo.ParallaxBranch.U0_MINUS],
                        datasets=all_datasets
                    )
                else:
                    self._log("Skipping extraction - using existing direct fits")
            else:
                # Use initial grids or existing fits
                need_grid = (self.n_loc > 1 and
                             not (state['has_parallax_fits'] and self._parallax_fits_match_n_loc()))

                if initial_grids is not None:
                    # Multi-location: extract and optimize minima from initial grids
                    self._log("\nExtracting and optimizing parallax solutions from initial grid")
                    self._extract_and_optimize_parallax_solutions(
                        initial_grids[mmexo.ParallaxBranch.U0_PLUS],
                        initial_grids[mmexo.ParallaxBranch.U0_MINUS],
                        datasets=all_datasets
                    )
                elif state['has_parallax_fits']:
                    # Re-optimize existing fits
                    self._log("\nRe-optimizing existing parallax fits")
                    self._reoptimize_existing_parallax_fits(datasets=all_datasets)
                # If n_loc==1 and no grid, direct fits already done in initial section

            self._save_restart_state()

        # ----------------------------------------------------------------
        # Execute workflow
        # ----------------------------------------------------------------
        fit_primary_location_if_needed()
        initial_grids = run_initial_parallax_if_needed()
        renormalize_if_needed()
        comprehensive_parallax_fitting(initial_grids)

    def fit_binary_lens(self) -> None:
        """
        Run binary-lens workflow, building on point-lens pieces.

        - static PSPL
        - Anomaly Finder search
        """
        # TODO: rework this workflow similar to how fit_point_lens was reworked.
        if self._datasets_changed:
            self._refit_models()

        # Reuse the shared pieces you actually need:
        self._ensure_static_point_lens()
        self._ensure_static_finite_point_lens()
        self._save_restart_state()

        # Now do binary-specific stuff:
        self._run_af_grid_search()
        self._fit_binary_models()
        self._save_restart_state()

    # ---------------------------------------------------------------------
    # Core helper: run_fit_if_needed
    # ---------------------------------------------------------------------

    def run_fit_if_needed(
            self,
            key: mmexo.FitKey,
            fit_func,
            datasets=None,
    ) -> mmexo.FitRecord:
        """
        Ensure there is an up-to-date mmexo.FitRecord for `key`.

        Parameters
        ----------
        key : mmexo.FitKey
            Which fit to run.
        fit_func : callable
            Function that runs the fit:
            `fit_func(initial_params: Optional[dict], datasets: Optional[list]) -> mmexo.MMEXOFASTFitResults`.
        datasets : list or None, optional
            Datasets to use for fitting. If None, uses self.datasets.

        Returns
        -------
        mmexo.FitRecord
            The current record for this fit (existing or newly fitted).
        """
        if datasets is None:
            datasets = self.datasets

        record = self.all_fit_results.get(key)

        # If we have a fixed or complete result, reuse it
        if record is not None and (record.fixed or record.is_complete):
            return record

        # Use existing params as a starting point, if present
        initial_params = record.params if record is not None else None

        # If no initial params, look for related fit to seed from
        if initial_params is None:
            related_fit = self._find_related_fit(key)
            if related_fit is not None:
                initial_params = related_fit.params
                self._log(f"Seeding from related fit: {mmexo.fit_types.model_key_to_label(related_fit.model_key)}")

        # Run the actual fit
        full_result = fit_func(initial_params=initial_params, datasets=datasets)

        # Derive renorm factors from current state, if any
        renorm_factors = self.renorm_factors

        new_record = mmexo.FitRecord.from_full_result(
            model_key=key,
            full_result=full_result,
            renorm_factors=renorm_factors,
            fixed=False,
        )
        self.all_fit_results.set(new_record)

        if self.verbose:
            self._save_restart_state()

        return new_record

    def _find_related_fit(self, key):
        """
        Find existing fit with same model type but different locations_used.

        Useful for seeding fits when location coverage changes.

        Parameters
        ----------
        key : mmexo.FitKey
            The fit key to find related fits for

        Returns
        -------
        FitRecord or None
            Best matching fit with same model but different locations, or None if not found
        """
        # Find all fits matching model type
        related_fits = []
        for existing_key, fit in self.all_fit_results.items():
            if (existing_key.lens_type == key.lens_type and
                    existing_key.source_type == key.source_type and
                    existing_key.parallax_branch == key.parallax_branch and
                    existing_key.lens_orb_motion == key.lens_orb_motion and
                    existing_key.locations_used != key.locations_used):  # Different locations

                related_fits.append((existing_key, fit))

        if len(related_fits) == 0:
            return None

        # Prefer most complete (by location count), then best chi2
        return max(related_fits, key=lambda x: (
            self._count_locations_used(x[0].locations_used),
            -x[1].chi2()
        ))[1]

    # ------------------------------------------------------------------
    # Shared point-lens steps
    # ------------------------------------------------------------------

    def _ensure_static_point_lens(self, datasets=None) -> None:
        """
        Make sure static PSPL exists in all_fit_results.

        Parameters
        ----------
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.
        """
        if datasets is None:
            datasets = self.datasets

        # Determine locations_used for the key
        locations_used = self._get_location_for_datasets(datasets)

        static_pspl_key = mmexo.FitKey(
            lens_type=mmexo.LensType.POINT,
            source_type=mmexo.SourceType.POINT,
            parallax_branch=mmexo.ParallaxBranch.NONE,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
            locations_used=locations_used,
        )

        def fit_static_pspl(initial_params=None, datasets=None):
            return self._fit_initial_pspl_model(initial_params=initial_params, datasets=datasets)

        self.run_fit_if_needed(static_pspl_key, fit_static_pspl, datasets=datasets)

    def _ensure_static_finite_point_lens(self, datasets=None) -> None:
        """
        Make sure static FSPL exists, if finite_source is enabled.

        Parameters
        ----------
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.
        """
        if not self.finite_source:
            return

        if datasets is None:
            datasets = self.datasets

        # Determine locations_used for the key
        locations_used = self._get_location_for_datasets(datasets)

        static_fspl_key = mmexo.FitKey(
            lens_type=mmexo.LensType.POINT,
            source_type=mmexo.SourceType.FINITE,
            parallax_branch=mmexo.ParallaxBranch.NONE,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
            locations_used=locations_used,
        )

        def fit_static_fspl(initial_params=None, datasets=None):
            return self._fit_static_fspl_model(initial_params=initial_params, datasets=datasets)

        self.run_fit_if_needed(static_fspl_key, fit_static_fspl, datasets=datasets)

    def _ensure_point_lens_parallax_models(self, datasets=None) -> None:
        """
        Make sure all configured point-lens parallax branches are fitted.

        Parameters
        ----------
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.
        """
        if datasets is None:
            datasets = self.datasets

        # Determine locations_used for the key
        locations_used = self._get_location_for_datasets(datasets)

        for par_key_base in self._iter_parallax_point_lens_keys():
            # Add locations_used to the key
            par_key = mmexo.FitKey(
                lens_type=par_key_base.lens_type,
                source_type=par_key_base.source_type,
                parallax_branch=par_key_base.parallax_branch,
                lens_orb_motion=par_key_base.lens_orb_motion,
                locations_used=locations_used,
            )

            def make_fit_func(k: mmexo.FitKey):
                def fit_func(initial_params=None, datasets=None):
                    return self._fit_pl_parallax_model(k, initial_params=initial_params, datasets=datasets)

                return fit_func

            self.run_fit_if_needed(par_key, make_fit_func(par_key), datasets=datasets)

    def _renormalize_location(self, reference_model, datasets_to_renorm, fit_datasets):
        """
        Renormalize errors for specific datasets using a reference model.

        Parameters
        ----------
        reference_model : MulensModel.Model
            Reference model for error renormalization
        datasets_to_renorm : list
            Datasets to renormalize (typically from one location)
        fit_datasets : list
            All datasets to include in the event for proper flux fitting
            (provides context for renormalization)
        """
        self._log(f"Renormalizing {len(datasets_to_renorm)} dataset(s)")

        # Calculate error factors for datasets_to_renorm only
        error_factors = self._remove_outliers_and_calc_errfacs(
            reference_model,
            fit_datasets=fit_datasets
        )

        # Apply renormalization (only to datasets_to_renorm)
        self._apply_error_renormalization(error_factors, datasets=datasets_to_renorm)

    def _reoptimize_existing_parallax_fits(self, datasets):
        """
        Re-optimize existing parallax fits with updated datasets.

        Takes existing parallax fit parameters and re-optimizes them
        with the current dataset collection.

        Parameters
        ----------
        datasets : list
            Datasets to use for re-optimization
        """
        # Get all parallax fits
        parallax_fits = []
        for key, record in self.all_fit_results.items():
            if key.parallax_branch != mmexo.ParallaxBranch.NONE:
                parallax_fits.append((key, record))

        if len(parallax_fits) == 0:
            self._log("Warning: No existing parallax fits to re-optimize")
            return

        self._log(f"Re-optimizing {len(parallax_fits)} parallax fit(s)")

        for key, record in parallax_fits:
            label = mmexo.fit_types.model_key_to_label(key)
            self._log(f"  Re-optimizing {label}")

            # Use existing params as starting point
            initial_params = record.params

            # Re-optimize
            fitter = mmexo.fitters.SFitFitter(
                initial_model_params=initial_params,
                datasets=datasets,
                **self._get_fitter_kwargs()
            )
            fitter.run()

            # Update record
            full_result = mmexo.MMEXOFASTFitResults(fitter)
            new_record = mmexo.FitRecord.from_full_result(
                model_key=key,
                full_result=full_result,
                renorm_factors=self.renorm_factors,
                fixed=False,
            )
            self.all_fit_results.set(new_record)

            self._log(f"    chi2 = {full_result.chi2:.2f}")

    def _parallax_fits_match_n_loc(self):
        """
        Check if existing parallax fits match current n_loc.

        Returns
        -------
        bool
            True if existing parallax branches are appropriate for current n_loc,
            False otherwise.
        """
        # Get existing parallax branches
        existing_branches = set(key.parallax_branch for key, _ in self.all_fit_results.items()
                                if key.parallax_branch != mmexo.ParallaxBranch.NONE)

        if len(existing_branches) == 0:
            return False

        # Determine expected branches for current n_loc
        if self.n_loc == 1:
            expected_branches = {mmexo.ParallaxBranch.U0_PLUS, mmexo.ParallaxBranch.U0_MINUS}
        else:
            expected_branches = {mmexo.ParallaxBranch.U0_PP, mmexo.ParallaxBranch.U0_PM,
                                 mmexo.ParallaxBranch.U0_MP, mmexo.ParallaxBranch.U0_MM}

        # Check if there's any overlap
        return bool(existing_branches & expected_branches)

    # ---------------------------------------------------------------------
    # Point-lens helpers:
    # ---------------------------------------------------------------------

    def _fit_initial_pspl_model(
            self,
            initial_params: Optional[Dict[str, float]] = None,
            datasets=None,
    ) -> mmexo.MMEXOFASTFitResults:
        """
        Estimate or accept starting point for PSPL, then run SFitFitter.

        EF grid is only used if `initial_params` is None and
        best_ef_grid_point is not yet available.

        Parameters
        ----------
        initial_params : dict or None, optional
            Starting parameters for fit
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.

        Returns
        -------
        mmexo.MMEXOFASTFitResults
            Fit results
        """
        if datasets is None:
            datasets = self.datasets

        if initial_params is None:
            if self.best_ef_grid_point is None:
                self.best_ef_grid_point = self.do_ef_grid_search()
                self._log(f"Best EF grid point {self.best_ef_grid_point}")

            pspl_est_params = mmexo.estimate_params.get_PSPL_params(
                self.best_ef_grid_point,
                datasets,
            )
            self._log(f"Initial PSPL Estimate {pspl_est_params}")
        else:
            pspl_est_params = initial_params
            self._log(f"Using initial PSPL params (user/previous): {pspl_est_params}")

        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=pspl_est_params, datasets=datasets, **self._get_fitter_kwargs())
        fitter.run()
        self._log(f'Initial SFit {fitter.best}')
        self._log_file_only(fitter.get_diagnostic_str())

        return mmexo.MMEXOFASTFitResults(fitter)

    def _fit_static_fspl_model(
            self,
            initial_params: Optional[Dict[str, float]] = None,
            datasets=None,
    ) -> mmexo.MMEXOFASTFitResults:
        """
        Fit a finite-source point-lens (FSPL) model.

        Parameters
        ----------
        initial_params : dict or None, optional
            Starting parameters for fit
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.

        Returns
        -------
        mmexo.MMEXOFASTFitResults
            Fit results
        """
        if datasets is None:
            datasets = self.datasets

        if initial_params is None:
            # Seed from static PSPL record if available
            static_pspl_key = mmexo.FitKey(
                lens_type=mmexo.LensType.POINT,
                source_type=mmexo.SourceType.POINT,
                parallax_branch=mmexo.ParallaxBranch.NONE,
                lens_orb_motion=mmexo.LensOrbMotion.NONE,
            )
            pspl_record = self.all_fit_results.get(static_pspl_key)
            if pspl_record is None:
                raise RuntimeError(
                    "Static PSPL must be fitted (or provided) before FSPL."
                )
            fspl_est_params = dict(pspl_record.params)
            fspl_est_params['rho'] = 1.5 * fspl_est_params['u_0']

        else:
            fspl_est_params = initial_params

        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=fspl_est_params, datasets=datasets, **self._get_fitter_kwargs())
        fitter.run()
        self._log(f'FSPL: {fitter.best}')
        self._log_file_only(fitter.get_diagnostic_str())

        return mmexo.MMEXOFASTFitResults(fitter)

    # --- parallax branch sign definitions and helpers ----------------

    BRANCH_SIGNS = {
        mmexo.ParallaxBranch.U0_PLUS: (+1, +1),
        mmexo.ParallaxBranch.U0_MINUS: (-1, -1),
        mmexo.ParallaxBranch.U0_PP: (+1, +1),
        mmexo.ParallaxBranch.U0_MM: (-1, -1),
        mmexo.ParallaxBranch.U0_PM: (+1, -1),
        mmexo.ParallaxBranch.U0_MP: (-1, +1),
    }

    def _apply_branch_signs(
            self,
            params: Dict[str, float],
            src_branch: mmexo.ParallaxBranch,
            target_branch: mmexo.ParallaxBranch,
    ) -> None:
        """
        In-place: adjust params from src_branch convention to target_branch.

        Flips signs of u_0 and/or pi_E_N as needed.

        Parameters
        ----------
        params : dict
            Parameter dictionary to modify in place
        src_branch : mmexo.ParallaxBranch
            Source parallax branch
        target_branch : mmexo.ParallaxBranch
            Target parallax branch
        """
        su0_src, spi_src = self.BRANCH_SIGNS[src_branch]
        su0_tgt, spi_tgt = self.BRANCH_SIGNS[target_branch]

        u0_factor = su0_tgt / su0_src
        piN_factor = spi_tgt / spi_src

        if "u_0" in params:
            params["u_0"] *= u0_factor
        if "pi_E_N" in params:
            params["pi_E_N"] *= piN_factor

    def _iter_parallax_point_lens_keys(self) -> Iterable[mmexo.FitKey]:
        """
        Yield mmexo.ModelKeys for all point-lens parallax models consistent with n_loc.

        Yields
        ------
        mmexo.FitKey
            Parallax model keys
        """
        if self.n_loc == 1:
            branches = [mmexo.ParallaxBranch.U0_PLUS, mmexo.ParallaxBranch.U0_MINUS]
        else:
            branches = [
                mmexo.ParallaxBranch.U0_PP,
                mmexo.ParallaxBranch.U0_MM,
                mmexo.ParallaxBranch.U0_PM,
                mmexo.ParallaxBranch.U0_MP,
            ]

        for branch in branches:
            yield mmexo.FitKey(
                lens_type=mmexo.LensType.POINT,
                source_type=(
                    mmexo.SourceType.FINITE if self.finite_source else mmexo.SourceType.POINT
                ),
                parallax_branch=branch,
                lens_orb_motion=mmexo.LensOrbMotion.NONE,
            )

    def _get_parallax_initial_params(
            self,
            key: mmexo.FitKey,
            initial_params: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Decide how to initialize parameters for a parallax point-lens fit.

        Priority:
        1. Use provided initial_params if not None.
        2. Seed from an existing parallax branch result, transformed via sign flips.
        3. Fallback to static point-lens params (PSPL/FSPL) for this source_type.

        Parameters
        ----------
        key : mmexo.FitKey
            Fit key for the parallax model
        initial_params : dict or None
            Provided initial parameters

        Returns
        -------
        dict
            Initial parameters for the fit
        """
        # 1. Caller-supplied initial params
        if initial_params is not None:
            self._log(
                f"Using provided initial params for parallax branch "
                f"{key.parallax_branch}: {initial_params}"
            )
            return dict(initial_params)

        # 2. Seed from any existing parallax branch
        for other_branch in self.BRANCH_SIGNS.keys():
            if other_branch == key.parallax_branch:
                continue

            other_key = mmexo.FitKey(
                lens_type=key.lens_type,
                source_type=key.source_type,
                parallax_branch=other_branch,
                lens_orb_motion=key.lens_orb_motion,
            )
            other_record = self.all_fit_results.get(other_key)
            if other_record is None:
                continue

            base = dict(other_record.params)
            self._apply_branch_signs(
                base,
                src_branch=other_branch,
                target_branch=key.parallax_branch,
            )
            self._log(
                f"Seeding parallax branch {key.parallax_branch.value} from "
                f"existing branch {other_branch.value} with transformed params: {base}"
            )
            return base

        # 3. Fallback: static point-lens
        static_key = mmexo.FitKey(
            lens_type=mmexo.LensType.POINT,
            source_type=key.source_type,
            parallax_branch=mmexo.ParallaxBranch.NONE,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
        )
        static_record = self.all_fit_results.get(static_key)
        if static_record is None:
            raise RuntimeError(
                "Static point-lens model must be available before parallax fits."
            )

        base = dict(static_record.params)
        base['pi_E_N'] = 0.
        base['pi_E_E'] = 0.
        self._log(
            f"Seeding parallax branch {key.parallax_branch.value} from "
            f"static model (source_type={key.source_type.value}): {base}"
        )
        return base

    def _fit_pl_parallax_model(
            self,
            key: mmexo.FitKey,
            initial_params: Optional[Dict[str, float]] = None,
            datasets=None,
    ) -> mmexo.MMEXOFASTFitResults:
        """
        Fit a point-lens parallax model for the given parallax branch.

        Parameters
        ----------
        key : mmexo.FitKey
            Fit key identifying the parallax model
        initial_params : dict or None, optional
            Starting parameters for fit
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.

        Returns
        -------
        mmexo.MMEXOFASTFitResults
            Fit results
        """
        if datasets is None:
            datasets = self.datasets

        par_est_params = self._get_parallax_initial_params(key, initial_params)

        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=par_est_params, datasets=datasets, **self._get_fitter_kwargs())
        fitter.run()
        self._log(f'{mmexo.fit_types.model_key_to_label(key)}: {fitter.best}')
        self._log_file_only(fitter.get_diagnostic_str())

        return mmexo.MMEXOFASTFitResults(fitter)

    def _run_piE_grid_search(self, datasets=None, grid_params=None, skip_optimization=False,
                             save_results=True, file_prefix='', file_suffix='',
                             refinement_params=None):
        """
        Run parallax grid search over pi_E_E and pi_E_N for U0_PLUS and U0_MINUS branches.

        For each branch, performs a grid search with adaptive refinement and optionally
        saves results and/or plots.
        Results are saved to files named: {file_head}{file_prefix}_piE_grid_{branch}{file_suffix}.txt
        Plot is saved as: {file_head}_piE_grid.png

        Parameters
        ----------
        datasets : list or None, optional
            Datasets to use. If None, uses self.datasets.
        grid_params : dict or None, optional
            Grid parameters (min, max, step for pi_E_E and pi_E_N).
            If None, uses self.PARALLAX_GRID_PARAMS_FINE.
        skip_optimization : bool, optional
            If True, calculate chi2 without optimization (faster, for coarse grids).
            Default is False.
        save_results : bool, optional
            If True, save grid results to file (when save_grid_results is configured).
            Default is True.
        file_prefix : str, optional
            Prefix to add before '_piE_grid' in filename. Default is ''.
        file_suffix : str, optional
            Suffix to add after branch name in filename. Default is ''.
        refinement_params : dict or None, optional
            Parameters for run_with_refinement(). If None, uses defaults:
            {'chi2_threshold': 200, 'min_step_size': 0.005, 'radius_steps': 2}
            Keys: chi2_threshold, min_step_size, radius_steps

        Returns
        -------
        dict
            Dictionary mapping ParallaxBranch to ParallaxGridSearch objects
        """
        if datasets is None:
            datasets = self.datasets

        if grid_params is None:
            grid_params = self.PARALLAX_GRID_PARAMS_FINE

        # Set default refinement parameters
        if refinement_params is None:
            refinement_params = {
                'chi2_threshold': 200,
                'min_step_size': 0.005,
                'radius_steps': 2
            }

        # Get reference model
        reference_fit = self._select_preferred_static_point_lens_model()
        reference_model = reference_fit.full_result.fitter.get_model()
        static_params = reference_model.parameters.parameters

        # Iterate over parallax branches
        branches = [mmexo.ParallaxBranch.U0_PLUS, mmexo.ParallaxBranch.U0_MINUS]

        grids = {}  # Store grid objects for plotting

        for branch in branches:
            self._log(f"Running piE grid search for {branch.value}")

            # Create and run grid search with refinement
            grid = mmexo.ParallaxGridSearch(
                datasets=datasets,
                static_params=static_params,
                grid_params=grid_params,
                fitter_kwargs=self._get_fitter_kwargs(),
                skip_optimization=skip_optimization,
                verbose=self.verbose,
            )
            grid.run_with_refinement(**refinement_params)

            grids[branch] = grid

            # Save results if configured AND requested
            if self.output.config.save_grid_results and save_results:
                filename = f"{self.output.config.file_head}{file_prefix}_piE_grid_{branch.value.lower()}{file_suffix}.txt"
                filepath = self.output.config.base_dir / filename
                grid.save_results(filepath, parallax_branch=branch.value)
                self._log(f"Saved grid results to {filepath}")

        # Create plot if configured
        if self.output.config.save_plots:
            self._plot_piE_grid_search(grids)

        return grids

    def _plot_piE_grid_search(self, grids):
        """
        Create 2-panel plot of piE grid search results.

        Parameters
        ----------
        grids : dict
            Dictionary mapping ParallaxBranch to ParallaxGridSearch objects
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Find global minimum chi2 for consistent coloring
        all_chi2 = []
        for grid in grids.values():
            all_chi2.extend([r['chi2'] for r in grid.results])

        min_chi2 = min(all_chi2)

        # Create figure with gridspec layout: 2 plots + colorbar
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.3)

        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
        cax = fig.add_subplot(gs[2])

        branches = [mmexo.ParallaxBranch.U0_PLUS, mmexo.ParallaxBranch.U0_MINUS]

        for i, (ax, branch) in enumerate(zip(axes, branches)):
            grid = grids[branch]

            # Plot grid points
            scatter = grid.plot_grid_points(ax=ax, min_chi2=min_chi2)

            # Formatting
            ax.set_xlabel(r'$\pi_{\rm E,E}$')
            ax.set_ylabel(r'$\pi_{\rm E,N}$')
            ax.set_title(branch.value)
            ax.invert_xaxis()
            ax.set_aspect('equal')
            ax.minorticks_on()

            # Turn off y-axis label and tick labels on right plot
            if i == 1:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)

        # Add colorbar
        fig.colorbar(scatter, cax=cax, label=r'$\sigma$ (min $\chi^2$ = ' + f'{min_chi2:.2f})')

        # Save plot
        self.output.save_plot('piE_grid', fig)

    def _get_best_from_grids(self, grids):
        """
        Get the best solution across multiple grid searches.

        Parameters
        ----------
        grids : dict
            Dictionary mapping ParallaxBranch to ParallaxGridSearch objects

        Returns
        -------
        tuple
            (chi2, params) of the best solution across all grids
        """
        best_overall = None
        best_chi2 = float('inf')

        for branch, grid in grids.items():
            best = np.nanargmin([r.get('chi2', np.nan) for r in grid.results])
            chi2 = grid.results[best].get('chi2')
            params = grid.results[best].get('params')
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_overall = (chi2, params)

        if best_overall is None:
            raise ValueError("No valid minima found in any grid")

        return best_overall

    # space parallax
    def _get_space_u0_sign(self, fit_result, space_ephemerides_file):
        """
        Determine the sign of u0 for the space observatory trajectory.

        Parameters
        ----------
        fit_result : FitRecord
            Fit result containing the model
        space_ephemerides_file : str
            Path to ephemerides file for the space observatory

        Returns
        -------
        str
            'P' for positive u0, 'M' for negative u0
        """
        params = fit_result.get_params_from_results()
        
        # Get model from fit_result
        model = MulensModel.Model(
            parameters=params,
            coords=self.coords,
            ephemerides_file=space_ephemerides_file
        )

        # Define function to minimize: u(t) = sqrt(x^2 + y^2)
        def u_squared(time):
            trajectory = model.get_trajectory([time])
            return trajectory.x[0] ** 2 + trajectory.y[0] ** 2

        # Find time of minimum u
        t_0 = params['t_0']
        t_E = params['t_E']

        result = minimize_scalar(
            u_squared,
            bounds=(t_0 - 2 * t_E, t_0 + 2 * t_E),
            method='bounded'
        )

        # Get y-coordinate at minimum
        t_min = result.x
        trajectory = model.get_trajectory([t_min])
        y_min = trajectory.y[0]

        # Return sign
        if y_min >= 0:
            return 'P'
        else:
            return 'M'

    def _fit_primary_location(self, primary_location=None, primary_dataset=None):
        """
        Fit primary location with static point lens models.

        Selects the primary location (automatically or by specification),
        fits static PSPL/FSPL, and renormalizes errors if configured.

        Parameters
        ----------
        primary_location : str or None, optional
            Location name to use as primary ('ground', 'Spitzer', etc.).
            If None, automatically selects location with longest time coverage.
        primary_dataset : str or None, optional
            Label of dataset to use for identifying primary location.
            Takes precedence over primary_location.

        Returns
        -------
        list
            Primary location datasets that were fit
        """
        # Select primary location datasets
        if primary_dataset is not None:
            primary_datasets = self._get_location_group_for_dataset(primary_dataset)
            # Find location name
            for loc, datasets in self.location_groups.items():
                if set(datasets) == set(primary_datasets):
                    self._primary_location = loc
                    break

            self._log(f"Using primary dataset: {primary_dataset} (location: {self._primary_location})")
        elif primary_location is not None:
            primary_datasets = self._get_location_group_by_name(primary_location)
            self._primary_location = primary_location
            self._log(f"Using primary location: {primary_location}")

        else:
            primary_datasets = self._select_primary_location_by_coverage()
            # Determine location name
            self._primary_location = None
            for loc, datasets in self.location_groups.items():
                if set(datasets) == set(primary_datasets):
                    self._primary_location = loc
                    break

            self._log(f"Auto-selected primary location: {self._primary_location}")

        # Fit static models with primary location only (skip parallax)
        self._log(f"Fitting static models with {len(primary_datasets)} primary location datasets")
        self.fit_point_lens(datasets=primary_datasets, skip_parallax=True)

        return primary_datasets

    def _add_location_and_grid_search(self, datasets, grid_params, skip_optimization=False,
                                      save_results=True, file_prefix='', file_suffix=''):
        """
        Run parallax grid search with specified datasets and grid parameters.

        Parameters
        ----------
        datasets : list
            Datasets to use for grid search
        grid_params : dict
            Grid parameters (pi_E_E_min, pi_E_E_max, pi_E_E_step, etc.)
        skip_optimization : bool, optional
            If True, skip parameter optimization (faster). Default is False.
        save_results : bool, optional
            If True, save grid results to file. Default is True.
        file_prefix : str, optional
            Prefix to add before '_piE_grid' in filename. Default is ''.
        file_suffix : str, optional
            Suffix to add after branch name in filename. Default is ''.

        Returns
        -------
        dict
            Dictionary mapping ParallaxBranch to ParallaxGridSearch objects
            Keys: mmexo.ParallaxBranch.U0_PLUS, mmexo.ParallaxBranch.U0_MINUS
        """
        locations_used = self._get_location_for_datasets(datasets)
        self._log(f"Running parallax grid search with locations: {locations_used}")

        # Run grid search and return the grids
        grids = self._run_piE_grid_search(
            datasets=datasets,
            grid_params=grid_params,
            skip_optimization=skip_optimization,
            save_results=save_results,
            file_prefix=file_prefix,
            file_suffix=file_suffix
        )

        return grids

    def _extract_and_optimize_parallax_solutions(self, u0_plus_grid, u0_minus_grid, datasets):
        """
        Extract minima from parallax grids and optimize each solution.

        For n_loc=2, extracts minima from both grids, optimizes each, determines
        u0 signs for both locations, and stores as PP/PM/MP/MM solutions.

        For n_loc>2, stores solutions as U0_PLUS or U0_MINUS based on primary
        location u0 sign only.

        Parameters
        ----------
        u0_plus_grid : ParallaxGridSearch
            Grid search results for U0_PLUS branch
        u0_minus_grid : ParallaxGridSearch
            Grid search results for U0_MINUS branch
        datasets : list
            Datasets to use for optimization (typically all available)
        """
        locations_used = self._get_location_for_datasets(datasets)

        # For n_loc=2, identify secondary location and get its ephemerides
        secondary_ephem = None
        if self.n_loc == 2:
            for loc, loc_datasets in self.location_groups.items():
                if loc != self._primary_location:
                    # Found secondary location
                    if len(loc_datasets) > 0:
                        # Get ephemerides file (could be None for ground)
                        secondary_ephem = getattr(loc_datasets[0], 'ephemerides_file', None)
                    break

            if secondary_ephem is None and self._primary_location != 'ground':
                # Secondary must be ground (no ephemerides)
                pass  # secondary_ephem stays None, which is correct for ground

        # Process both grids in a loop
        grid_configs = [
            (u0_plus_grid, mmexo.ParallaxBranch.U0_PLUS, 'U0_PLUS'),
            (u0_minus_grid, mmexo.ParallaxBranch.U0_MINUS, 'U0_MINUS')
        ]

        for grid, base_branch, grid_name in grid_configs:
            # Extract minima from this grid
            minima = grid.find_local_minima()
            self._log(f"Found {len(minima)} minima in {grid_name} grid")

            # Optimize each minimum
            for i, (chi2, params) in enumerate(minima):
                self._log(f"Optimizing {grid_name} minimum {i + 1}/{len(minima)}")

                # Optimize with full datasets
                fit_result = self._optimize_parallax_solution(params, datasets)

                # Determine branch based on n_loc
                if self.n_loc == 2:
                    # Check secondary location u0 sign
                    secondary_sign = self._get_space_u0_sign(fit_result, secondary_ephem)

                    # Map to PP/PM/MP/MM based on primary (base_branch) and secondary signs
                    if base_branch == mmexo.ParallaxBranch.U0_PLUS:
                        branch = mmexo.ParallaxBranch.U0_PP if secondary_sign == 'P' else mmexo.ParallaxBranch.U0_PM
                    else:  # U0_MINUS
                        branch = mmexo.ParallaxBranch.U0_MP if secondary_sign == 'P' else mmexo.ParallaxBranch.U0_MM
                else:
                    # n_loc > 2: just use base branch (U0_PLUS or U0_MINUS)
                    branch = base_branch

                # Create FitKey
                fit_key = mmexo.FitKey(
                    lens_type=mmexo.LensType.POINT,
                    source_type=mmexo.SourceType.FINITE if self.finite_source else mmexo.SourceType.POINT,
                    parallax_branch=branch,
                    lens_orb_motion=mmexo.LensOrbMotion.NONE,
                    locations_used=locations_used,
                )

                # Check if this key already exists (edge case warning)
                if fit_key in self.all_fit_results:
                    self._log(f"WARNING: FitKey {branch.value} already exists in all_fit_results. "
                                    f"This may indicate multiple minima with the same u0 sign combination. "
                                    f"Overwriting previous result.")

                # Create and store FitRecord
                record = mmexo.FitRecord.from_full_result(
                    model_key=fit_key,
                    full_result=fit_result,
                    renorm_factors=self.renorm_factors,
                    fixed=False,
                )
                self.all_fit_results.set(record)
                self._log(f"Stored {branch.value} solution (chi2={fit_result.chi2:.2f})")

    def _optimize_parallax_solution(self, initial_params, datasets):
        """
        Optimize a parallax solution starting from grid parameters.

        Parameters
        ----------
        initial_params : dict
            Starting parameters from grid search minimum
        datasets : list
            Datasets to use for optimization

        Returns
        -------
        MMEXOFASTFitResults
            Optimized fit results
        """
        self._log(f"Optimizing from u_0{'+' if initial_params.get('u_0') >= 0 else '-'} grid point: "
                        f"pi_E_E={initial_params.get('pi_E_E', 'N/A'):.3f}, "
                        f"pi_E_N={initial_params.get('pi_E_N', 'N/A'):.3f}")

        # Run SFitFitter with all parameters free
        fitter = mmexo.fitters.SFitFitter(
            initial_model_params=initial_params,
            datasets=datasets,
            **self._get_fitter_kwargs()
        )
        fitter.run()

        self._log(f"Optimized: chi2={fitter.best.get('chi2'):.2f}, {fitter.best}")

        return mmexo.MMEXOFASTFitResults(fitter)

    def _select_best_static_pspl(self) -> Optional[mmexo.FitRecord]:
        """
        Select the best static PSPL model for current datasets.

        Priority order:
        1. Exact match to current locations_used
        2. Partial match including primary_location
        3. Partial match not including primary_location
        4. Any static PSPL

        Among same priority, prefers lowest chi2.

        Returns
        -------
        FitRecord or None
            Best static PSPL fit, or None if not found
        """
        # Get current locations
        current_locations_used = self._get_location_for_datasets(self.datasets)
        current_locs_set = set(current_locations_used.split('+')) if current_locations_used else set()

        # Find all static PSPL fits
        pspl_fits = []
        for key, fit in self.all_fit_results.items():
            if (key.lens_type == mmexo.LensType.POINT and
                    key.source_type == mmexo.SourceType.POINT and
                    key.parallax_branch == mmexo.ParallaxBranch.NONE and
                    key.lens_orb_motion == mmexo.LensOrbMotion.NONE):

                chi2 = fit.chi2()
                if chi2 is not None:
                    pspl_fits.append((key, fit, chi2))

        if len(pspl_fits) == 0:
            return None

        # Categorize by priority
        def get_priority(key):
            """Return (priority, chi2) for sorting. Lower priority number = higher priority."""
            fit_locs = key.locations_used

            # Exact match
            if fit_locs == current_locations_used:
                return (0, None)

            # Parse fit locations
            if fit_locs is None:
                fit_locs_set = set()
            else:
                fit_locs_set = set(fit_locs.split('+'))

            # Check for partial match
            if fit_locs_set & current_locs_set:  # Has overlap
                if self._primary_location in fit_locs_set:
                    return (1, None)  # Includes primary
                else:
                    return (2, None)  # Doesn't include primary

            # No match
            return (3, None)

        # Sort by priority, then chi2
        pspl_fits.sort(key=lambda x: (get_priority(x[0])[0], x[2]))

        return pspl_fits[0][1]

    def _select_best_parallax_pspl(self) -> Optional[mmexo.FitRecord]:
        """
        Select the best parallax PSPL model for current datasets.

        Only considers fits that match current n_loc. Among those:

        Priority order:
        1. Exact match to current locations_used
        2. Partial match including primary_location
        3. Partial match not including primary_location
        4. Any parallax PSPL matching n_loc

        Among same priority, prefers lowest chi2.

        Returns
        -------
        FitRecord or None
            Best parallax PSPL fit, or None if not found
        """
        # Get current locations
        current_locations_used = self._get_location_for_datasets(self.datasets)
        current_locs_set = set(current_locations_used.split('+')) if current_locations_used else set()

        # Find all parallax PSPL fits matching current n_loc
        parallax_fits = []
        for key, fit in self.all_fit_results.items():
            if (key.lens_type == mmexo.LensType.POINT and
                    key.source_type == mmexo.SourceType.POINT and
                    key.parallax_branch != mmexo.ParallaxBranch.NONE and
                    key.lens_orb_motion == mmexo.LensOrbMotion.NONE):

                # Check if parallax branch matches n_loc
                if self.n_loc == 1:
                    expected_branches = {mmexo.ParallaxBranch.U0_PLUS, mmexo.ParallaxBranch.U0_MINUS}
                else:
                    expected_branches = {mmexo.ParallaxBranch.U0_PP, mmexo.ParallaxBranch.U0_PM,
                                         mmexo.ParallaxBranch.U0_MP, mmexo.ParallaxBranch.U0_MM}

                if key.parallax_branch not in expected_branches:
                    continue  # Skip fits for wrong n_loc

                chi2 = fit.chi2()
                if chi2 is not None:
                    parallax_fits.append((key, fit, chi2))

        if len(parallax_fits) == 0:
            return None

        # Categorize by priority
        def get_priority(key):
            """Return (priority, chi2) for sorting. Lower priority number = higher priority."""
            fit_locs = key.locations_used

            # Exact match
            if fit_locs == current_locations_used:
                return (0, None)

            # Parse fit locations
            if fit_locs is None:
                fit_locs_set = set()
            else:
                fit_locs_set = set(fit_locs.split('+'))

            # Check for partial match
            if fit_locs_set & current_locs_set:  # Has overlap
                if self._primary_location in fit_locs_set:
                    return (1, None)  # Includes primary
                else:
                    return (2, None)  # Doesn't include primary

            # No match
            return (3, None)

        # Sort by priority, then chi2
        parallax_fits.sort(key=lambda x: (get_priority(x[0])[0], x[2]))

        return parallax_fits[0][1]

    # other parallax helpers
    def _select_preferred_static_point_lens_model(self, chi2_threshold=20):
        """
        Select the preferred static point lens model from self.all_fit_results.

        Prefers models with more complete location coverage. Among models with
        the same location coverage, prefers FSPL over PSPL if chi2 improvement
        exceeds chi2_threshold.

        Parameters
        ----------
        chi2_threshold : float, optional
            Minimum chi2 improvement required to prefer FSPL over PSPL.
            Default is 20.

        Returns
        -------
        FitRecord
            The preferred static point lens fit result

        Raises
        ------
        ValueError
            If no static point lens models exist
        """
        # Find all PSPL and FSPL fits regardless of locations_used
        pspl_fits = []
        fspl_fits = []

        for key, fit in self.all_fit_results.items():
            if (key.lens_type == mmexo.LensType.POINT and
                    key.parallax_branch == mmexo.ParallaxBranch.NONE and
                    key.lens_orb_motion == mmexo.LensOrbMotion.NONE):

                if key.source_type == mmexo.SourceType.POINT:
                    pspl_fits.append((key, fit))
                elif key.source_type == mmexo.SourceType.FINITE:
                    fspl_fits.append((key, fit))

        # Check if at least one exists
        if len(pspl_fits) == 0 and len(fspl_fits) == 0:
            raise ValueError("No static point lens models found in all_fit_results")

        # Select most complete version of each type
        def get_most_complete(fits_list):
            if len(fits_list) == 0:
                return None
            # Sort by location completeness (descending), then by chi2 (ascending)
            return max(fits_list, key=lambda x: (
                self._count_locations_used(x[0].locations_used),
                -x[1].chi2()  # Negative for ascending chi2
            ))[1]

        pspl_fit = get_most_complete(pspl_fits)
        fspl_fit = get_most_complete(fspl_fits)

        # If only one type exists, return it
        if pspl_fit is None:
            return fspl_fit
        if fspl_fit is None:
            return pspl_fit

        # Both exist - compare chi2
        pspl_chi2 = pspl_fit.chi2()
        fspl_chi2 = fspl_fit.chi2()

        # Return FSPL only if significantly better
        if fspl_chi2 < pspl_chi2 - chi2_threshold:
            return fspl_fit
        else:
            return pspl_fit

    def _select_preferred_point_lens(
            self,
            delta_chi2_threshold: float = 50.0,
    ) -> Optional[mmexo.FitRecord]:
        """
        Choose the preferred PSPL model for the binary workflow.

        Policy:
        - If any parallax models exist, pick the best parallax model.
        - Compare its chi^2 to the best static PL chi^2.
        - If (chi2_static - chi2_parallax) > delta_chi2_threshold,
            → use parallax model.
          Else
            → use static PL model.
        - If no parallax models exist, fall back to static PL.
        - If neither exists (or no chi^2), return None.

        Parameters
        ----------
        delta_chi2_threshold : float, optional
            Minimum chi2 improvement for parallax to be preferred. Default is 50.

        Returns
        -------
        FitRecord or None
            Preferred point lens model
        """
        best_static = self._select_preferred_static_point_lens_model()
        best_par = self._select_best_parallax_pspl()  # Changed from self.all_fit_results.select_best_parallax_pspl()

        chi2_static = best_static.chi2() if best_static is not None else None
        chi2_par = best_par.chi2() if best_par is not None else None

        # Case 1: no parallax available or no parallax chi^2
        if chi2_par is None:
            return best_static

        # Case 2: no static available or no static chi^2 → default to parallax
        if chi2_static is None:
            return best_par

        # Case 3: both exist; apply threshold rule
        improvement = chi2_static - chi2_par
        if improvement > delta_chi2_threshold:
            # parallax is significantly better
            return best_par
        else:
            # static is as good or better (within threshold)
            return best_static

    # ---------------------------------------------------------------------
    # Binary-lens helpers:
    # ---------------------------------------------------------------------
    def _run_af_grid_search(self):
        """
        Run Anomaly Finder grid search if not already done.
        """
        if self.best_af_grid_point is None:
            self.best_af_grid_point = self.do_af_grid_search()
            self._log(f'Best AF grid {self.best_af_grid_point}')

        if self.anomaly_lc_params is None:
            self.anomaly_lc_params = self.get_anomaly_lc_params()
            self._log(f'Anomaly Params {self.anomaly_lc_params}')

    def _fit_binary_models(self):
        """
        Fit binary lens models (currently only wide planet in GG97 limit).

        Raises
        ------
        NotImplementedError
            Binary fitting only partially implemented
        """

        def fit_wide_planet():
            wide_planet_fitter = mmexo.fitters.WidePlanetFitter(
                datasets=self.datasets, anomaly_lc_params=self.anomaly_lc_params,
            )
            wide_planet_fitter.estimate_initial_parameters()
            self._log(
                f'Initial 2L1S Wide Model {wide_planet_fitter.initial_model}' +
                f'\nmag methods {wide_planet_fitter.mag_methods}')

            wide_planet_fitter.run()
            self._log_file_only(wide_planet_fitter.get_diagnostic_str())
            return wide_planet_fitter.best

        fit_wide_planet()
        raise NotImplementedError('fitting binary models only partially implemented')

    # ---------------------------------------------------------------------
    # Data helpers:
    # ---------------------------------------------------------------------
    def set_residuals(self, pspl_params):
        """
        Calculate and store residuals from a PSPL model.

        Parameters
        ----------
        pspl_params : dict
            PSPL model parameters
        """
        event = MulensModel.Event(
            datasets=self.datasets, model=MulensModel.Model(pspl_params))
        event.fit_fluxes()
        residuals = []
        for i, dataset in enumerate(self.datasets):
            res, err = event.fits[i].get_residuals(phot_fmt='flux')
            residuals.append(
                MulensModel.MulensData(
                    [dataset.time, res, err], phot_fmt='flux',
                    bandpass=dataset.bandpass,
                    ephemerides_file=dataset.ephemerides_file))

        self.residuals = residuals

# ---------------------------------------------------------------------
    # Renormalization helpers:
    # ---------------------------------------------------------------------
    #def renormalize_errors_and_refit(
    #        self,
    #        reference_model,
    #        datasets=None,
    #):
    #    """
    #    Renormalize photometric errors and refit all models.
    #
    #    Parameters
    #    ----------
    #    reference_model : MulensModel.Model
    #        The model to use as reference for error renormalization.
    #        Can be obtained from a FitRecord via
    #        FitRecord.full_result.get_model()
    #    datasets : list or None, optional
    #        List of datasets to process. If None, use all datasets.
    #
    #    Returns
    #    -------
    #    list
    #        Updated list of dataset objects after renormalization.
    #        If datasets=None was passed, returns self.datasets.
    #    """
    #    if datasets is None:
    #        datasets = self.datasets
    #        return_all = True
    #    else:
    #        return_all = False
    #        # Track labels of input datasets
    #        input_labels = [ds.plot_properties['label'] for ds in datasets]
    #
    #    # Renormalize errors using the reference model
    #    error_factors = self._remove_outliers_and_calc_errfacs(
    #        reference_model,
    #        fit_datasets=datasets
    #    )
    #    self._apply_error_renormalization(error_factors)
    #
    #    # Refit all models with renormalized errors
    #    self._refit_models()
    #
    #    # Return updated dataset objects
    #    if return_all:
    #        return self.datasets
    #    else:
    #        # Map labels back to new dataset objects
    #        label_to_dataset = {ds.plot_properties['label']: ds
    #                           for ds in self.datasets}
    #        return [label_to_dataset[label] for label in input_labels]

    def _remove_outliers_and_calc_errfacs(self, reference_model, fit_datasets=None):
        """
        Remove outliers and calculate error renormalization factors.

        Parameters
        ----------
        reference_model : mmexo.Model
            Model to use for outlier detection
        fit_datasets : list or None, optional
            Datasets to fit. If None, uses self.datasets.

        Returns
        -------
        dict
            Dictionary mapping label to error renormalization factor
            for each dataset that was processed.
        """
        if fit_datasets is None:
            fit_datasets = self.datasets

        # Determine which datasets need processing
        datasets_to_process = [
            dataset for dataset in fit_datasets
            if dataset.plot_properties['label'] not in self.renorm_factors
        ]

        if not datasets_to_process:
            self._log("All datasets already have renormalization factors applied.")
            return {}

        self._log(f'Initial reference model: \n{reference_model}')
        # Create event with ALL datasets for proper flux fitting
        event = MulensModel.Event(
            datasets=fit_datasets, model=reference_model, coords=self.coords)
        event.fit_fluxes()

        self._log("Starting outlier removal...")

        error_factors_dict = {}

        # Process only the specified datasets
        for dataset in datasets_to_process:
            # Find index in fit_datasets
            if dataset not in fit_datasets:
                raise ValueError(f"Dataset {dataset} not found in fit_datasets")

            i = fit_datasets.index(dataset)
            dataset_name = dataset.plot_properties.get('label', f'Dataset {i}')
            self._log(f"\nProcessing {dataset_name}:")

            bad_index = -1
            n_good = np.sum(dataset.good)
            n_params = len(reference_model.parameters.as_dict())
            found_bad = []

            # Iteratively remove outliers
            while (bad_index is not None) and (n_good > 0):
                event.fit_fluxes()

                n_good = np.sum(dataset.good)
                dof = n_good - n_params

                if dof <= 0:
                    self._log(f"  Warning: dof={dof}, stopping outlier removal")
                    break

                # Calculate significance threshold
                max_sig = np.max([np.sqrt(2) * erfcinv(1. / dof), 3])

                # Get chi2 and error factor
                chi2 = event.get_chi2_for_dataset(i)
                errfac = np.sqrt(chi2 / dof)

                self._log_file_only(f"  errfac={errfac:.3f}, n_good={n_good}, dof={dof}")

                # Get residuals and calculate sigma
                (res, err) = event.fits[i].get_residuals(phot_fmt='flux', bad=True)
                sigma = np.abs(res / (err * errfac))

                # Find outliers
                n_still_bad = np.sum(sigma[dataset.good] > max_sig)

                if n_still_bad > 0:
                    # Find and mark the worst point
                    i_worst = np.argmax(sigma[dataset.good])
                    bad_index = np.argwhere(sigma == sigma[dataset.good][i_worst])[0]

                    new_bad = dataset.bad.copy()
                    new_bad[bad_index] = True
                    dataset.bad = new_bad

                    found_bad.append(bad_index[0])
                    self._log_file_only(
                        f"  Marked point {bad_index[0]} as bad: "
                        f"n_bad={np.sum(dataset.bad)}, n_good={np.sum(dataset.good)}"
                    )
                else:
                    bad_index = None

            # Calculate final error factor
            event.fit_fluxes()
            final_chi2 = event.get_chi2_for_dataset(i)
            final_dof = np.sum(dataset.good) - n_params

            if final_dof > 0:
                final_errfac = np.sqrt(final_chi2 / final_dof)
            else:
                final_errfac = 1.0

            error_factors_dict[dataset.plot_properties['label']] = final_errfac

            # Summary
            if len(found_bad) > 0:
                self._log(f"  Removed {len(found_bad)} outliers, errfac={final_errfac:.3f}")
            else:
                self._log(f"  No outliers removed, errfac={final_errfac:.3f}")

        return error_factors_dict

    def _apply_error_renormalization(self, error_factors, datasets=None):
        """
        Recreate datasets with renormalized errors.

        Parameters
        ----------
        error_factors : dict
            Dictionary mapping label to error renormalization factor
        datasets : list or None, optional
            Datasets to renormalize. If None, renormalizes all datasets
            that have labels in error_factors.
        """
        if datasets is None:
            # Apply to all datasets that have factors
            datasets = [ds for ds in self.datasets
                        if ds.plot_properties['label'] in error_factors]

        if not datasets:
            self._log("No datasets to renormalize.")
            return

        self._log("\nApplying error renormalization...")

        # Get the signature of MulensData.__init__
        sig = inspect.signature(MulensModel.MulensData.__init__)

        new_datasets = []
        for dataset in datasets:
            # Get error factor for this dataset
            label = dataset.plot_properties['label']
            errfac = error_factors.get(label)
            if errfac is None:
                self._log(f"Warning: No error factor for {label}, skipping")
                continue

            # Build kwargs dict from original object's attributes
            kwargs = {}
            for param_name in sig.parameters:
                if param_name in ['self', 'data_list', 'good', 'phot_fmt', 'file_name']:
                    continue

                if hasattr(dataset, param_name):
                    kwargs[param_name] = getattr(dataset, param_name)

            # Create new dataset with scaled errors
            new_dataset = MulensModel.MulensData(
                data_list=[dataset.time, dataset.flux, errfac * dataset.err_flux],
                phot_fmt='flux',
                **kwargs
            )

            self._log(new_dataset)
            new_datasets.append(new_dataset)

        # Build mapping old -> new and replace in self.datasets
        old_to_new = dict(zip(datasets, new_datasets))
        self.datasets = [old_to_new.get(ds, ds) for ds in self.datasets]

        # Update flux fixing maps with new dataset objects
        self.fix_blend_flux_map = self._map_label_dict_to_datasets(self.fix_blend_flux)
        self.fix_source_flux_map = self._map_label_dict_to_datasets(self.fix_source_flux)

        # Store applied factors in state
        self.renorm_factors.update(error_factors)

        self._log("Datasets recreated with renormalized errors")

    def _refit_models(self):
        """
        Refit all models using current datasets and previous fit results as
        initial parameters.

        Updates all_fit_results in place with new fit results.
        """
        self._log("\nUpdating fits...")
        for key, fit_record in self.all_fit_results.items():
            # Get the fitter object
            fitter = fit_record.full_result.fitter

            # Update with current (potentially renormalized) datasets
            fitter.datasets = self.datasets

            # Use previous fit as starting point
            fitter.initial_model_params = fit_record.params

            # Refit
            fitter.run()

            full_result = mmexo.MMEXOFASTFitResults(fitter)
            new_record = mmexo.FitRecord.from_full_result(
                model_key=key,
                full_result=full_result,
                fixed=False,
            )
            self.all_fit_results.set(new_record)

            self._log(f'{mmexo.fit_types.model_key_to_label(key)}: {fitter.best}')
            self._log_file_only(fitter.get_diagnostic_str())

# ---------------------------------------------------------------------
    # External search helpers:
    # ---------------------------------------------------------------------
    def do_ef_grid_search(self):
        """
        Run a EventFinderGridSearch.

        Returns
        -------
        dict
            Best EventFinder grid point parameters
        """
        ef_grid = mmexo.EventFinderGridSearch(datasets=self.datasets)
        ef_grid.run()

        if self.output is not None and self.output.config.save_plots:
            fig = ef_grid.plot()
            self.output.save_plot('ef_grid', fig)

        return ef_grid.best

    def do_af_grid_search(self):
        """
        Run an AnomalyFinderGridSearch.

        Returns
        -------
        dict
            Best AnomalyFinder grid point parameters
        """
        self.set_residuals(self._select_best_static_pspl().params)
        af_grid = mmexo.AnomalyFinderGridSearch(residuals=self.residuals)
        # May need to update value of teff_min
        af_grid.run()
        return af_grid.best

    def get_anomaly_lc_params(self):
        """
        Estimate anomaly light curve parameters.

        Returns
        -------
        dict
            Anomaly light curve parameters
        """
        estimator = mmexo.estimate_params.AnomalyPropertyEstimator(
            datasets=self.datasets, pspl_params=self._select_best_static_pspl().params,
            af_results=self.best_af_grid_point)
        return estimator.get_anomaly_lc_parameters()

    # ---------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        """
        Log message to console/file based on verbose/save_log settings.

        Parameters
        ----------
        msg : str
            Message to log
        """
        if self.output is not None:
            self.output.log(msg)
        elif self.verbose:
            # Fallback: print to console if no output manager but verbose=True
            print(msg)

    def _log_file_only(self, msg: str) -> None:
        """
        Log message to file only (never console).

        Parameters
        ----------
        msg : str
            Message to log to file
        """
        if self.output is not None and self.output.logger is not None:
            self.output.logger.info(msg)

    def _output_latex_table(self, name: str = 'results', models=None) -> None:
        """
        Output LaTeX table of results.

        Parameters
        ----------
        name : str, optional
            Table name. Default is 'results'.
        models : list or None, optional
            Models to include in table. If None, includes all.
        """
        if self.output is not None:
            table_str = self.make_ulens_table(table_type='latex', models=models)
            self.output.save_latex_table(name, table_str)

    def _save_restart_state(self) -> None:
        """Save current state for restarting fits."""
        if self.output is None:
            return

        restart_data = {
            'config': self._get_config(),
            'state': self._get_state(),
        }

        state_bytes = pickle.dumps(restart_data)
        self.output.save_restart_state(state_bytes)

    def make_ulens_table(self, table_type: Optional[str], models=None) -> str:
        """
        Return a string consisting of a formatted table summarizing the results
        of the microlensing fits.

        Parameters
        ----------
        table_type : str or None
            'ascii' (default) or 'latex'.
        models : list, optional
            - None: include all models in self.all_fit_results
            - list of labels (str): e.g., ['PSPL static', 'PSPL par u0+']
            - list of mmexo.ModelKey: explicit selection.

        Returns
        -------
        str
            Table in the requested format.
        """

        def order_df(df: pd.DataFrame) -> pd.DataFrame:
            """
            Order parameters in a human-friendly way (ulens params first,
            then fluxes).

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame to order

            Returns
            -------
            pd.DataFrame
                Ordered DataFrame
            """

            def get_ordered_ulens_keys_for_repr(n_sources: int = 1):
                """
                Define the default order of microlensing parameters.

                Parameters
                ----------
                n_sources : int, optional
                    Number of sources. Default is 1.

                Returns
                -------
                list
                    Ordered list of parameter names
                """
                basic_keys = ["t_0", "u_0", "t_E", "rho", "t_star"]
                additional_keys = [
                    "pi_E_N", "pi_E_E", "t_0_par", "s", "q", "alpha",
                    "convergence_K", "shear_G", "ds_dt", "dalpha_dt", "s_z",
                    "ds_z_dt", "t_0_kep",
                    "x_caustic_in", "x_caustic_out", "t_caustic_in", "t_caustic_out",
                    "xi_period", "xi_semimajor_axis", "xi_inclination",
                    "xi_Omega_node", "xi_argument_of_latitude_reference",
                    "xi_eccentricity", "xi_omega_periapsis", "q_source", "t_0_xi",
                ]

                ordered_keys: list[str] = []
                if n_sources > 1:
                    for param_head in basic_keys:
                        if param_head == "t_E":
                            ordered_keys.append(param_head)
                        else:
                            for i in range(n_sources):
                                ordered_keys.append(f"{param_head}_{i + 1}")
                else:
                    ordered_keys = list(basic_keys)

                ordered_keys.extend(additional_keys)

                # New for MMEXOFAST:
                ordered_keys = ["chi2", "N_data"] + ordered_keys

                return ordered_keys

            def get_ordered_flux_keys_for_repr() -> list[str]:
                """
                Get ordered list of flux parameter names.

                Returns
                -------
                list
                    Ordered list of flux parameter names
                """
                flux_keys: list[str] = []
                for i, dataset in enumerate(self.datasets):
                    if "label" in dataset.plot_properties.keys():
                        obs = dataset.plot_properties["label"].split("-")[0]
                    else:
                        obs = i

                    if dataset.bandpass is not None:
                        band = dataset.bandpass
                    else:
                        band = "mag"

                    flux_keys.append(f"{band}_S_{obs}")
                    flux_keys.append(f"{band}_B_{obs}")

                return flux_keys

            def get_ordered_keys_for_repr() -> list[str]:
                """
                Get complete ordered list of parameter names.

                Returns
                -------
                list
                    Ordered list of all parameter names
                """
                ulens_keys = get_ordered_ulens_keys_for_repr()
                flux_keys = get_ordered_flux_keys_for_repr()
                return ulens_keys + flux_keys

            desired_order = get_ordered_keys_for_repr()
            order_map = {name: i for i, name in enumerate(desired_order)}

            df["sort_key"] = df["parameter_names"].map(order_map)
            df["orig_pos"] = range(len(df))

            # Anything not in desired_order goes to the end
            max_key = len(desired_order)
            df["sort_key"] = df["sort_key"].fillna(max_key)
            df = (
                df.sort_values(["sort_key", "orig_pos"])
                    .reset_index()
                    .drop(columns=["index", "sort_key", "orig_pos"])
            )
            return df

        if table_type is None:
            table_type = "ascii"

        # Normalize `models` to a list of (label, mmexo.FitRecord)
        model_label_record_pairs: list[tuple[str, mmexo.FitRecord]] = []

        if models is None:
            # All models currently in mmexo.AllFitResults
            for key, record in self.all_fit_results.items():
                label = mmexo.fit_types.model_key_to_label(key)
                model_label_record_pairs.append((label, record))
        else:
            for m in models:
                if isinstance(m, mmexo.FitKey):
                    key = m
                else:
                    # assume string label
                    key = mmexo.fit_types.label_to_model_key(m)
                record = self.all_fit_results.get(key)
                if record is None:
                    raise ValueError(f"No mmexo.FitRecord found for model {m!r}")
                label = mmexo.fit_types.model_key_to_label(key)
                model_label_record_pairs.append((label, record))

        results_table: Optional[pd.DataFrame] = None

        for label, record in model_label_record_pairs:
            new_column = record.to_dataframe()
            new_column = new_column.rename(
                columns={
                    "values": label,
                    "sigmas": f"sig [{label}]",
                }
            )

            if results_table is None:
                results_table = new_column
            else:
                results_table = results_table.merge(
                    new_column,
                    on="parameter_names",
                    how="outer",
                )

        if results_table is None:
            # No models; return empty table
            return ""

        results_table = order_df(results_table)

        if table_type == "latex":
            def fmt(name: str) -> str:
                """
                Format parameter name for LaTeX.

                Parameters
                ----------
                name : str
                    Parameter name

                Returns
                -------
                str
                    LaTeX formatted name
                """
                if name == "chi2":
                    return r"$\chi^2$"

                parts = name.split("_")
                if len(parts) == 1:
                    return f"${name}$"
                first = parts[0]
                rest = ", ".join(parts[1:])
                return f"${first}" + "_{" + rest + "}$"

            results_table["parameter_names"] = results_table["parameter_names"].apply(
                fmt
            )
            return results_table.to_latex(index=False)

        elif table_type == "ascii":
            with pd.option_context(
                    "display.max_rows", None,
                    "display.max_columns", None,
                    "display.width", None,
                    "display.float_format", "{:f}".format,
            ):
                return results_table.to_string(index=False)

        else:
            raise NotImplementedError(table_type + " not implemented.")

    # ---------------------------------------------------------------------
    # EXOZIPPy Helpers
    # ---------------------------------------------------------------------
    def initialize_exozippy(self):
        """
        Get the best-fit microlensing parameters for initializing exozippy fitting.

        Returns
        -------
        dict
            Dictionary with keys:
                'fits': list of dict
                    [{'parameters': {dict of ulens parameters},
                      'sigmas': {dict of uncertainties in ulens parameters}} ...]
                'errfacs': list of error renormalization factors for each dataset.
                    DEFAULT: None
                'mag_methods': list of magnification methods following the MulensModel
                    convention. DEFAULT: None
        """
        initializations = {'fits': [], 'errfacs': None, 'mag_methods': None}

        if self.fit_type == 'point lens':
            fits = []
            for par_key in self._iter_parallax_point_lens_keys():
                fits.append({'parameters': self.all_fit_results.get(par_key).params,
                             'sigmas': self.all_fit_results.get(par_key).sigmas})

            initializations['fits'] = fits
        else:
            raise NotImplementedError('initialize_exozippy only implemented for point lens fits')

        return initializations


### Project Context:
I have an existing class: MMEXOFAST. The vision for the class is that it will have the option for a one-touch work flow, but also that the user should be able to exit at any point, add more data, and resume. There were several possible workflows depending on different settings. This was quasi-implemented, but it doesn't work. We built a set of specs to think about the implementation from scratch. Now we need to start implementing while comparing to the existing code and seeing if anything can be reused. We want to have good unit test coverage. The idea is to work top-down to verify that the logic is correct and code up the unit tests as we go.

---

## OUTPUT

Provide:

1. **Main entry method pseudocode** (formatted as above)
2. **Data flow diagram** (text-based description of how data moves through steps)
3. **Complexity assessment** (which steps are [SIMPLE] and which are [COMPLEX]?)
4. **Assumptions** (what are you assuming about the spec or architecture?)
5. **Questions for clarification** (anything unclear that would change the flow?)

---

## NEXT STEPS

Once you've provided the above, I will review it with you and ask clarifying questions 
until we're confident this high-level flow is correct. Only then will we proceed to 
Phase 2.

**Do not proceed to the next phase until you explicitly confirm you're satisfied 
with this high-level flow.**
```

---


**Claude 4.6 Sonnet**
{"type":"think","think":"The user wants me to sketch the highest-level flow for the `fit()` method of `MMEXOFASTFitter`, based on the specifications provided. Let me carefully read through the specs and existing code to understand what the main entry point should do.\n\nThe main entry point is `fit()`. Let me trace through what it needs to do based on the specs:\n\nFrom Section 5 (Workflow Definitions):\n\n**R5.1.1** At the start of every `fit()` call, before any workflow steps execute:\n1. `_mark_stale_fits()` must be called (R3.1.7)\n2. If `workflow_stage` is `\"complete\"`, the R2.15 settings-change check must be performed\n\n\n3. New locations must be detected and the appropriate dataset addition workflow triggered (R3.1.8)\n\n**R5.1.2** Route to `fit_point_lens()` when `fit_type='point lens'`\n**R5.1.3** Route to `fit_binary_lens()` when `fit_type='binary lens'`\n**R5.1.4** Both must execute only remaining stages from current `workflow_stage`\n**R5.1.5** At end of every `fit()` call:\n1. `_check_renorm_completeness()` must be called\n2. `_output_latex_table()` must be called\n\nIf `fit_type` is None, I need to raise a ValueError immediately. Now I'm sketching out the overall structure of the fit method—starting with validation, logging the current workflow stage, marking stale fits, handling the complete stage with settings checks, detecting new locations for dataset additions, routing to the appropriate lens fitting function, and finally running the post-workflow cleanup steps. Looking at the complexity breakdown, validation is straightforward, but marking stale fits involves calling a helper method that could be more involved.\n\nThe data flow is becoming clearer—fit() doesn't receive input data directly but instead pulls from self.datasets, which means I need to understand how the datasets are structured and accessed throughout the workflow. I'm working through the assumptions about how the workflow state persists and gets restored. The key question is how the settings-change check knows what was used in the previous run—it must be comparing the current configuration against what was saved in the restart state. I also need to clarify what happens with the new location detection step and how it fits into the overall flow. I'm working through some clarification questions about the implementation details. The settings-change check needs to store and retrieve previous settings state, likely as part of the restart configuration. For the complete stage with settings changes, I need to determine whether restart state should be saved after each modification. The execution order is clear: `_mark_stale_fits()` runs first, then the settings-change check if complete, then new location detection—but I'm realizing that when settings reset the stage, those already-marked stale records may need additional handling.\n\nDrafting the pseudocode structure...\n\nThen I need to handle the \"complete\" stage, which checks if settings have changed and either resets the workflow or marks records as needing refit. After that, I'm detecting new location datasets—if any appear, I bypass the forward check and trigger a different-location workflow, though there's complexity around how this interacts with binary versus point-source fits.\n\nNext comes the routing step where I call either fit_point_lens() or fit_binary_lens() based on the fit type and current stage. Finally, I'm handling post-workflow output by logging warnings for unrenormalized datasets and generating the LaTeX table.\n\nBut I'm reconsidering step 5—the new location detection actually triggers a single-pass different-location workflow that includes fitting itself, so it's not just a detection step but part of the actual workflow execution. So the flow is: new locations get detected in `fit()`, the stage resets to the pre-alt-location entry point, and then when `fit_point_lens` or `fit_binary_lens` runs, it picks up from that reset stage and executes the alt-location sub-workflow as part of its normal flow. R3.2.2 describes what happens when those new datasets are encountered—the single-pass workflow executes to handle them. fitting a static PSPL, coarse parallax grid, etc. This is what `_run_alt_location_subworkflow()` does.\n\nSo the new location detection in step 5 of fit() just:\n1. Detects new locations\n2. Resets the stage (bypassing forward-check)\n3. The actual workflow execution happens in fit_point_lens/fit_binary_lens\n\nThis makes sense because the stage reset is what tells fit_point_lens/fit_binary_lens to execute the alt-location stages.\n\nActually, hmm. But R3.1.8 also says: \"For `fit_type='binary lens'` and `workflow_stage < 'post_binary_renorm_done'`, newly detected non-primary locations are not processed yet — they will be handled after the binary stage.\" This suggests the detection happens in fit(), but the actual processing is deferred based on the current stage.\n\nSo the logic is: when new locations are detected, the stage only resets if we're either doing point lens fitting or already past the binary stage. For binary lens fits still in progress, the newly detected locations get queued up and processed later once we reach the post-binary stage. The settings-change logic is more nuanced than I initially thought—depending on what changed, it either resets the stage to a specific point or does nothing, and then the subsequent routing picks up from wherever the stage lands. I'm now writing out the full pseudocode for the main `fit()` function to capture this flow clearly.\n\nWriting the fit function pseudocode... log(f\"fit() called. Current workflow_stage: {self.workflow_stage}\")\n\nI'm marking stale fits first, which checks all existing results against renormalization factors and location groups to flag anything that needs refitting, while being careful never to touch fixed records. This might reset the workflow stage backward depending on what's stale.\n\nNow I'm handling the case where we're already at the \"complete\" stage but settings have changed—I need to check if fit_type, finite_source, parallax_grid, or renormalize_errors have been modified since the last run, and depending on which one changed, I either reset to an earlier stage, trigger an immediate parallax grid run, or mark all fits for refitting.\n\nAfter that, I'm detecting any new locations that have appeared in the current data but weren't present in any of the previous fit records, and if I find them, I reset the stage appropriately based on whether we're doing power-law or binary lens fitting, being careful to respect the timing constraints of the binary workflow.\n\nThen I route to the appropriate fitting sub-workflow based on the fit type, letting each method skip stages it's already completed, and finally I check whether renormalization was fully applied across all datasets and write out the LaTeX results table. Looking at the complexity breakdown, steps 3, 4, and 5 are where the real work happens—marking stale fits, handling config changes, and detecting new locations all require careful orchestration. I'm assuming the settings-change check will need access to whatever config was saved during the restart, either still in memory or pulled from the restart file.\n\nThe key insight is that these steps have dependencies: stale fits get marked first, which might reset the workflow stage before the settings check even runs. And if the stage is already reset, the settings check won't trigger at all since it only runs when workflow_stage is \"complete\". For new locations, I'm looking for any location in self.location_groups that hasn't appeared in any existing FitRecord yet. The parallel execution model seems to be cut off, but the sequencing of these operations matters—stale marking happens before the other checks, which determines what actually executes downstream. ) or should the sub-workflows simply be no-ops because all stages are already past? This affects whether we need an explicit early return. I'm also wondering about step 5 and new location detection—if the binary lens workflow is in progress and a new location shows up but we haven't reached the post_binary_renorm_done stage yet, should I store that detection somewhere so it doesn't get re-detected on every fit() call? And then there's the question of where R3.2.1 (same-location dataset addition with the two-pass workflow) actually gets handled—is it in _mark_stale_fits, in the new location detection step, or inside the fit functions themselves, since same-location additions do make fits stale but there's more to it than that.\n\nLooking at step 4, when parallax_grid changes to True, does _handle_complete_stage_settings_check run the grid search inline and then continue as a no-op, or should that be handled differently? I think I need to be clearer about what belongs in fit() versus what belongs in fit_point_lens and fit_binary_lens. The spec says _mark_stale_fits() must be called at the start of every fit() call before any workflow steps execute, so I should trace through exactly what that means for the orchestration. The two-pass workflow for same-location dataset additions is triggered when new datasets are detected from a location already in the fit records—the detection happens in step 3, and _mark_stale_fits already marks those fits for refitting as part of Pass 1. The full two-pass logic (refitting in Pass 1, then outlier detection and renormalization in Pass 2) appears to be embedded within the renormalization stage logic of the fit_point_lens workflow. the logical sequence of what happens, even if it's spread across multiple methods. The \"two-pass\" nature emerges from the workflow structure.\n\nSo the key is that new datasets can be identified by their absence from self.renorm_factors—they haven't been renormalized yet—which lets the renormalization stages distinguish between old and new data when running outlier removal and error factor calculations. This means the two-pass logic is actually built into fit_point_lens and fit_binary_lens during the renormalization stages, not something fit() needs to orchestrate explicitly.\n\nFor new locations, the fit() method detects them, resets the stage backward with force=True, and then the alt-location workflow runs inside the lens-fitting methods. So my pseudocode structure is on the right track—I just need to refine it and check the \"complete\" no-op case.\n\nWhen the workflow is already at \"complete\" with no changes, it stays at \"complete\" through step 4, and the lens-fitting methods naturally skip all stages since they're already past them—no explicit early return needed. But if _mark_stale_fits resets the stage from \"complete\" to an earlier state, then step 4's settings check won't run, and the sub-workflow picks up from that new stage instead. For the parallax_grid change case in R2.15, I need to trace through what \"run final parallax grid only\" means... I need to clarify whether the parallax grid runs for just the best point-lens model or across all parallax branches in the R2.15 case, and I'm also noticing that the spec doesn't fully specify how finite_source behaves when a binary fit exists. There's also an ambiguity about what happens when fit() is called a second time without changes and no restart file.\n\nLooking at this more carefully, the settings-change comparison really only matters when loading a restart file where the workflow is already complete—if the user just calls fit() again in the same session after it finished, the config hasn't changed anyway. But I should think through whether there's a meaningful \"previous state\" to compare against when no restart file was loaded. I'm noticing the spec starts numbering at 3 instead of 1, which suggests items 1 and 2 are defined elsewhere in R5.1.1. Looking back at the full requirement, it seems the numbering is just continuing from earlier steps in that same section, so I should treat 3, 4, 5 as the complete sequence for what needs to happen at the start of `fit()`.\n\nNow I need to carefully map out the post-workflow section since R5.1.5 specifies the order of `_check_renorm_completeness()` and `_output_latex_table()`, and R6.3.3 adds another constraint about when these get called relative to each other. I'm seeing some redundancy in the spec—_check_renorm_completeness gets called in multiple places, including within _output_latex_table itself. I'll follow R5.1.5 for the main fit() flow and note that the table method also invokes it internally. Now I'm organizing my response with the pseudocode, a data flow diagram, complexity analysis, and some clarifying questions.\n\nBuilding the data flow diagram... After step 5, the workflow stage determines which sub-workflow to route to—either fitting point lenses or binary lenses—which then updates the fit results, workflow stage, and renormalization factors. Finally, I'm checking renormalization completeness and outputting the LaTeX table as the post-workflow step.\n\nNow I'm identifying the key questions I need to clarify: how the settings comparison in step 4 detects previous configurations, whether the early return for unchanged settings happens before or within the sub-workflows, where the two-pass logic for same-location additions gets orchestrated, and how the parallax grid change factors into the completeness check. I'm noticing the interface specification doesn't match the implementation requirements—`fit_point_lens()` needs to accept an optional `stop_at` parameter according to R5.3.1a, but the spec shows it taking no arguments. I need to clarify how this parameter flows through the call chain, especially when `fit_binary_lens()` calls `fit_point_lens(stop_at=\"primary_renorm_done\")` versus the plain `fit_point_lens()` call in `fit()`.\n\nWriting out the full response with pseudocode... True, marking those records and resetting the workflow stage accordingly. Then I need to handle the \"complete\" stage case—checking whether any configuration settings have changed since the last run, which might trigger a conversion from PL to binary fit type.\n\nNow I'm working through what happens when settings do change: resetting the stage to \"primary_renorm_done\" if renormalization is enabled, or to \"static_fit_done\" if the finite source changed, marking all fits for refitting. If the parallax grid is enabled, I run that immediately without changing the stage. There's also the case where nothing changed and the workflow is essentially a no-op. I need to clarify how the binary fit presence affects the finite_source logic.\n\nFor step 5, I'm detecting locations that exist in the location_groups but haven't appeared in any of the fit records. If new locations are found and we're either doing point lens fitting or already past the binary renormalization stage, I reset the stage accordingly without marking existing fits for refitting. The logic branches differently for binary lens fits depending on what stage we're currently at.\n\nThen in step 6, I route to the appropriate sub-workflow based on the fit type and current stage, where each method skips any stages we've already completed. If we've already reached the \"complete\" stage by this point, the sub-workflow becomes a no-op since there's nothing left to process.\n\nAfter the workflow runs, I need to check whether renormalization was fully applied across all datasets, logging a warning if only some labels got renormalized. Finally, I'm generating the LaTeX output table from all the fit results. Now I'm mapping out the data flow through the validation steps—fit_type gets checked first, then workflow_stage is logged, followed by processing all_fit_results and renorm_factors through the pipeline. After marking stale fits, I check if the workflow stage is complete and handle any necessary updates or parallax grid runs. Then I move into detecting new locations by examining the location groups and fit record labels, which might force a workflow reset if changes are found.\n\nNow I'm routing to the appropriate sub-workflow based on the fit type—either point lens or binary lens fitting—which generates the final fit results, updates the workflow stage, and computes renormalization factors. Once those are complete, I wrap up with post-workflow checks on renormalization completeness and output the results as a LaTeX table.\n\nI'm realizing this workflow is getting complex, so I should step back and assess the actual complexity of each step to see where the real bottlenecks are. The \"complete\" state acts as a no-op when there are no changes—the workflow stages are already past, so the fit methods skip through without doing anything. The two-pass logic for same-location workflows is actually embedded within the individual fit methods themselves, not orchestrated at the top level, with staleness marking handling the first pass and renormalization stages handling the second. For binary workflows that detect new locations before reaching the post-renorm stage, the detection is deferred without needing explicit tracking.\n\nOn the next `fit()` call after binary stages finish, the new location will be detected again since no existing FitRecords will have it in their labels. A settings-change check only matters if a restart file was loaded between calls; otherwise, the saved config matches the current one and nothing changes. If `_mark_stale_fits()` already reset the workflow stage away from \"complete\", the settings-change condition won't trigger anyway, which is the right behavior.\n\nNow I'm working through some clarifying questions about the implementation. The main one is whether `_saved_config` gets populated from the restart file during initialization, or if I need a separate snapshot that updates each time the stage reaches \"complete\". I'm also wondering whether hitting the no-op case should return early from `fit()` or let the sub-workflows naturally become no-ops, and there's a third question about the parallax grid that got cut off. I'm verifying the execution order matches the spec: mark stale fits, then the settings-change check if we're at complete, then new location detection, then route to the appropriate workflow. That's the sequence I have laid out. Now I need to clarify what \"triggered\" means in R3.2.2 when the single-pass different-location workflow runs after a stage reset—whether it's just a stage transition or something more involved.\n\nLooking at the code flow, `_run_alt_location_subworkflow()` gets called from `fit_point_lens()` or `fit_binary_lens()` after the stage resets to \"primary_renorm_done\". From there, the system checks for ordered alternate locations and executes the alt-location sub-workflow for each one if any exist. So resetting to \"primary_renorm_done\" is what actually triggers the workflow to proceed through those stages naturally.\n\nNow I'm double-checking the stage reset logic in step 4 against R2.15 to make sure I have the conditions right.\n\nLooking at the different cases: when `fit_type` changes, the stage resets to `\"primary_renorm_done\"`; when `finite_source` changes, it resets to `\"static_fit_done\"` and marks fits for refitting; `parallax_grid` becoming `True` runs the final grid without resetting; and if `renormalize_errors` flips to `True`, the stage goes back to `\"pl_fit_done\"`. The key insight is that the \"no settings changed and no stale fits\" case in R2.15 is actually impossible to reach in step 4, because if stale fits were detected earlier in step 3, the stage would've already been reset away from `\"complete\"`, so step 4 wouldn't even run. condition in R2.15 is just clarifying when the no-op applies. I think I've been overanalyzing this—let me move forward with writing the response. I'm planning to structure it with the main entry method in pseudocode, a text-based data flow diagram, a complexity assessment table, a numbered list of assumptions, and then my clarifying questions all at once so we can work through them together."}


**Claude 4.6 Sonnet**
## 1. Main Entry Method Pseudocode

```python
def fit() -> None:
    """
    Main entry point for the MMEXOFAST fitting workflow.

    Validates configuration, runs pre-workflow checks (staleness detection,
    settings-change detection, new-location detection), routes to the appropriate
    sub-workflow based on fit_type, and generates post-fit output.

    Executes only stages that remain from the current workflow_stage —
    already-completed stages are skipped automatically.
    """

    # -----------------------------------------------------------------------
    # PRE-FLIGHT: Validate
    # -----------------------------------------------------------------------

    # Step 1: Validate fit_type  [SIMPLE]
    #   - Takes:    self.fit_type
    #   - Produces: raises ValueError if None; nothing otherwise
    #   - Per:      R5.1.6

    # Step 2: Log current workflow_stage  [SIMPLE]
    #   - Takes:    self.workflow_stage
    #   - Produces: log entry "fit() called. Current stage: <workflow_stage>"
    #   - Per:      R6.1.2
    #   - Note:     must fire BEFORE any state changes

    # -----------------------------------------------------------------------
    # PRE-FLIGHT: Staleness and settings checks
    # -----------------------------------------------------------------------

    # Step 3: Mark stale fits  [COMPLEX — delegates to _mark_stale_fits()]
    #   - Takes:    self.all_fit_results
    #               self.renorm_factors
    #               self.location_groups
    #   - Produces:
    #       * Stale FitRecords get needs_refit = True (records kept, params preserved as seeds)
    #       * self.workflow_stage reset backward to earliest stage implied by what was marked
    #         (see R3.1.5 rules: static marked → "event_search_done"; parallax only →
    #          "static_fit_done"; binary only → "primary_renorm_done"; alt-loc only →
    #          ("alt_loc_added", [...locs_excluding_stale...]))
    #       * Log: which fits were marked, reason (renorm change or same-loc dataset added),
    #              new stage
    #   - Delegates to: _get_stale_keys(), FitRecord.is_stale()
    #   - Per:      R3.1.4–R3.1.7, R6.1.3

    # Step 4: Handle "complete" stage — settings-change check
    #         [COMPLEX — delegates to _handle_complete_stage_settings_check()]
    #   - Condition: only runs if self.workflow_stage == "complete"
    #     (if Step 3 reset the stage, this step is skipped entirely)
    #   - Takes:    self.<config attributes>  vs  self._saved_config
    #   - Produces (one of the following scenarios):
    #       * fit_type changed (PL → binary):
    #           → stage reset to "primary_renorm_done" (force=True)
    #       * finite_source changed:
    #           → stage reset to "static_fit_done" (force=True)
    #           → all existing non-fixed fits marked needs_refit=True
    #           → [OPEN] nuanced if a binary fit exists; see Q5
    #       * parallax_grid newly True:
    #           → run final parallax grid immediately
    #           → stage NOT reset; remains "complete"
    #           → [OPEN] where exactly does this run? see Q3
    #       * renormalize_errors newly True:
    #           → stage reset to "pl_fit_done" (force=True)
    #           → all existing non-fixed fits marked needs_refit=True
    #       * No settings changed AND no stale fits:
    #           → log "workflow already complete", no state changes
    #           → sub-workflow (Step 6) will be a natural no-op
    #   - Per:      R2.15
    #   - Risk:     multiple scenarios, each with different reset behavior;
    #               finite_source nuance is partially unspecified

    # Step 5: Detect new locations and prepare for alt-location workflow
    #         [COMPLEX — delegates to _detect_and_handle_new_locations()]
    #   - Takes:    self.location_groups
    #               FitRecord.location_labels from all existing records
    #   - Produces:
    #       * Identifies locations present in self.location_groups but absent
    #         from every existing FitRecord.location_labels
    #       * If new locations exist AND conditions permit processing now:
    #           (fit_type='point lens')  OR
    #           (fit_type='binary lens' AND stage >= "post_binary_renorm_done")
    #           → stage reset to "primary_renorm_done" (PL) or
    #             "post_binary_renorm_done" (BL) using _set_stage(force=True)
    #           → existing fits are NOT marked needs_refit
    #           → sub-workflow (Step 6) will execute the alt-location stages
    #             for each new location via _run_alt_location_subworkflow()
    #       * If new locations exist AND fit_type='binary lens'
    #         AND stage < "post_binary_renorm_done":
    #           → defer: no action now; new locations processed after binary stages complete
    #           → [OPEN] bookkeeping for deferred locations; see Q4
    #   - Per:      R3.1.8, R3.2.2

    # -----------------------------------------------------------------------
    # WORKFLOW ROUTING
    # -----------------------------------------------------------------------

    # Step 6: Route to sub-workflow  [SIMPLE]
    #   - Takes:    self.fit_type
    #               self.workflow_stage (as updated by steps 3–5)
    #   - Produces:
    #       * Calls fit_point_lens() if fit_type == 'point lens'
    #       * Calls fit_binary_lens() if fit_type == 'binary lens'
    #       * Both methods skip stages already past workflow_stage
    #       * If workflow_stage is still "complete" after steps 3–5
    #         (e.g., R2.15 no-op), the sub-workflow executes zero stages
    #   - Per:      R5.1.2, R5.1.3, R5.1.4

    # -----------------------------------------------------------------------
    # POST-WORKFLOW OUTPUT
    # -----------------------------------------------------------------------

    # Step 7a: Check renormalization completeness  [SIMPLE]
    #   - Takes:    self.renorm_factors, self.datasets
    #   - Produces: warning logged listing unrenormalized dataset labels
    #               (only if some-but-not-all datasets are in renorm_factors)
    #   - Per:      R5.1.5, R3.7.2, R6.1.6

    # Step 7b: Output LaTeX table  [SIMPLE — delegates to _output_latex_table()]
    #   - Takes:    self.all_fit_results, self.datasets
    #   - Produces: LaTeX file written; skipped silently if all_fit_results is empty
    #               (internally calls _check_renorm_completeness again per R6.3.3)
    #               needs_refit=True and is_complete=False fits visually flagged in table
    #   - Per:      R5.1.5, R6.3.1–R6.3.4
```

---

## 2. Data Flow

```
                         ┌─────────────────────────────────────────────┐
  PERSISTENT STATE       │  self.fit_type                               │
  (set in __init__)      │  self.workflow_stage                         │
                         │  self.all_fit_results  (FitRecords)          │
                         │  self.renorm_factors                         │
                         │  self.location_groups  (property)            │
                         │  self._saved_config    (from restart file)   │
                         └────────────────────────┬────────────────────┘
                                                  │ (read at start of fit())
                                                  ▼
    ┌─ Step 1 ──────────────────────────────────────────────────────────┐
    │  validate fit_type → raises ValueError or passes through          │
    └───────────────────────────────────────────────────────────────────┘
                                                  │
    ┌─ Step 2 ──────────────────────────────────────────────────────────┐
    │  log: "Current stage: <workflow_stage>"                           │
    └───────────────────────────────────────────────────────────────────┘
                                                  │
    ┌─ Step 3: _mark_stale_fits() ──────────────────────────────────────┐
    │  reads:  all_fit_results, renorm_factors, location_groups         │
    │  writes: all_fit_results.needs_refit flags                        │
    │  writes: workflow_stage  (may reset backward)                     │
    │  writes: log                                                      │
    └───────────────────────────────────────────────────────────────────┘
                                                  │
              workflow_stage updated ─────────────┤
                                                  │  (only if still "complete")
    ┌─ Step 4: _handle_complete_stage() ────────────────────────────────┐
    │  reads:  current config attrs  vs  _saved_config                  │
    │  writes: workflow_stage  (may reset with force)                   │
    │  writes: all_fit_results.needs_refit  (for some scenarios)        │
    │  runs:   parallax grid (for parallax_grid change scenario)        │
    └───────────────────────────────────────────────────────────────────┘
                                                  │
    ┌─ Step 5: _detect_and_handle_new_locations() ──────────────────────┐
    │  reads:  location_groups                                          │
    │  reads:  FitRecord.location_labels  (from all_fit_results)        │
    │  writes: workflow_stage  (may reset backward with force)          │
    │  note:   does NOT write needs_refit                               │
    └───────────────────────────────────────────────────────────────────┘
                                                  │
              workflow_stage (final) ─────────────┤
              fit_type ───────────────────────────┤
                                                  ▼
    ┌─ Step 6: route ───────────────────────────────────────────────────┐
    │  fit_point_lens()  OR  fit_binary_lens()                          │
    │     reads:  workflow_stage, all_fit_results, datasets             │
    │     writes: all_fit_results (new FitRecords added)                │
    │     writes: workflow_stage (advances stage by stage)              │
    │     writes: renorm_factors, best_ef/af_grid_point, etc.          │
    └───────────────────────────────────────────────────────────────────┘
                                                  │
    ┌─ Step 7a: _check_renorm_completeness() ───────────────────────────┐
    │  reads:  renorm_factors, datasets                                 │
    │  writes: log (warning if partially renorm'd)                      │
    └───────────────────────────────────────────────────────────────────┘
                                                  │
    ┌─ Step 7b: _output_latex_table() ─────────────────────────────────┐
    │  reads:  all_fit_results (final state), datasets                  │
    │  writes: LaTeX file on disk                                       │
    └───────────────────────────────────────────────────────────────────┘
```

---

## 3. Complexity Assessment

| Step | What it does | Complexity | Why |
|---|---|---|---|
| 1 | Validate `fit_type` | **SIMPLE** | Single `if None` guard |
| 2 | Log `workflow_stage` | **SIMPLE** | Single `_log()` call |
| 3 | `_mark_stale_fits()` | **COMPLEX** | Must iterate all records, delegate to `FitRecord.is_stale()`, map stale fit types to the correct stage reset rule (R3.1.5 has 4 sub-cases), log clearly |
| 4 | `_handle_complete_stage_settings_check()` | **COMPLEX** | Five distinct response scenarios (R2.15), some involving stage resets + needs_refit marking, one involving running a grid search inline; finite_source case has binary-dependent nuance |
| 5 | `_detect_and_handle_new_locations()` | **COMPLEX** | Must compare location sets across `location_groups` and `FitRecord.location_labels`, handle binary deferral condition, use `force=True` stage bypass correctly |
| 6 | Route to sub-workflow | **SIMPLE** | A single `if/elif` — all complexity lives inside `fit_point_lens` / `fit_binary_lens` |
| 7a | `_check_renorm_completeness()` | **SIMPLE** | One method call; the method itself has straightforward logic |
| 7b | `_output_latex_table()` | **SIMPLE** | One method call; internal complexity is already in the existing implementation |

---

## 4. Assumptions

**A1 — Settings comparison mechanism.**
I'm assuming `self._saved_config` is populated in `__init__` from the restart file's `config` dict (it is already loaded by `_load_restart_data`). Step 4 compares each relevant config key against `self._saved_config.get(key)`. If no restart file was loaded, `_saved_config` is `{}`, so no settings-change is detected — which is correct behavior (no previous state to compare against).

**A2 — "Complete" no-op doesn't need an explicit early return.**
When step 4 detects no changes (R2.15 no-op), `workflow_stage` stays `"complete"`. When step 6 routes to `fit_point_lens` / `fit_binary_lens`, those methods will skip every stage (all are already past). The no-op emerges naturally from the skip logic rather than requiring an explicit `return` in `fit()`.

**A3 — Same-location two-pass workflow is not orchestrated by `fit()`.**
R3.2.1's two-pass structure is an emergent property of the existing workflow stages, not something `fit()` explicitly orchestrates. Step 3 (`_mark_stale_fits`) handles Pass 1 step 2 (marking needs_refit). Pass 1's refitting and Pass 2's renorm+refit happen inside `fit_point_lens` / `fit_binary_lens` at the appropriate renormalization stages. `fit()` doesn't need to know about "passes" explicitly.

**A4 — Deferred new-location detection requires no bookkeeping.**
When a new location is detected in step 5 but deferred (binary workflow not yet at `post_binary_renorm_done`), no explicit tracking is needed. On the next `fit()` call after the binary stages complete, step 5 will re-detect the new location because no `FitRecord` with that location's `location_labels` was created during the binary stages. The detection logic is idempotent.

**A5 — Step 4 only runs when stage is still `"complete"` after step 3.**
If `_mark_stale_fits()` resets the stage (e.g., to `"static_fit_done"`), step 4's condition fails and the settings-change check is skipped. This is intentional: the stale fits already determined what needs re-running; the settings-change check would be redundant and potentially conflicting.

**A6 — `workflow_stage == "complete"` after step 5 means the sub-workflow is a no-op.**
If steps 3–5 do not change the stage (e.g., R2.15 no-op and no new locations), `fit_point_lens` / `fit_binary_lens` will find every stage already past and return without doing work. Steps 7a and 7b still run unconditionally per R5.1.5.

---

## 5. Questions for Clarification

I have five questions. I'll number them by priority; **Q1 is the one I'd most like to resolve first** before we validate the rest of the flow.

---

**Q1 (Settings comparison — highest priority):**
Step 4 compares current config against a "previous" config. My assumption (A1) is that `self._saved_config` holds the config dict loaded from the restart file. But there's a gap: if the user calls `fit()` twice in the same session (no restart file, no state reload between calls), the "previous config" would be whatever was in `__init__` — meaning no changes would be detected, which is correct. However, *after* a complete run, the config was saved to the restart file. On the *next* session, it gets loaded back. This means the comparison is between the config at `__init__` time (current call) and the config from the restart file (previous call). 

Is this the intended comparison? Or should there be a dedicated `self._settings_snapshot` attribute that is explicitly updated each time `_set_stage("complete")` is called?

---

**Q2 (Parallax grid in the "complete" settings check):**
R2.15 says "parallax_grid changed to True: run final parallax grid only, do not reset stage." Where should this grid run execute? Two options:

- **A:** Inside `_handle_complete_stage_settings_check()` — runs inline, stage stays "complete", then sub-workflow is a no-op.
- **B:** Set a flag (e.g., `self._force_parallax_grid = True`), keep stage at "complete", and let `fit_point_lens`/`fit_binary_lens` check the flag and run the grid before returning.

Option A is simpler but mixes grid execution into the settings-check method. Option B is more consistent with the sub-workflow encapsulation.

---

**Q3 (Early return vs. natural no-op for "complete"):**
When R2.15 detects "no changes and no stale fits," the spec says "treat as no-op, log that workflow is already complete." Should `fit()` **return early** at that point (before step 5 and step 6), or should it continue through all remaining steps with the sub-workflow naturally doing nothing?

An early return would also skip steps 7a and 7b — which might be intentional (no output needed) or a problem (output might be desired even when no work was done).

---

**Q4 (Deferred new-location bookkeeping in binary workflow):**
Per R3.1.8, when `fit_type='binary lens'` and a new location is detected while `stage < "post_binary_renorm_done"`, it's deferred. My assumption (A4) is that no bookkeeping is needed because the detection is re-run each `fit()` call. 

Is this correct? Or is there a scenario where a `FitRecord` *with that location in `location_labels`* could be created during the binary stages (e.g., a static PSPL fit using all datasets) that would cause the new location to be "forgotten" before the binary workflow completes?

---

**Q5 (finite_source nuance at "complete"):**
R2.15 notes: "finite_source change has nuanced behavior depending on whether a binary fit exists." This appears intentionally unresolved. Should I treat this as a known gap and leave a `# TODO: binary + finite_source re-entry` placeholder for now, or is there a decision you've already made that I should incorporate?

---

**Ready to iterate.** I'd like to start with **Q1** since the answer determines whether `_handle_complete_stage_settings_check()` needs a dedicated attribute or can rely on the existing restart-file mechanism.


**User**
I have some questions too: 

For the steps marked COMPLEX, there are many lines of pseudo-code. But actually, those will just be a single line of code because they get delegated to other methods. Is that correct?


**Claude 4.6 Sonnet**
{"type":"think","think":"The user is making a very good point. They're asking whether the COMPLEX steps in `fit()` itself are just single method calls — because all the complexity lives in the delegated methods, not in `fit()` itself.\n\nLet me think about this carefully.\n\nLooking at the pseudocode I wrote:\n\n- Step 3 (`_mark_stale_fits()`) — yes, in `fit()` this is literally `self._mark_stale_fits()`. The complexity lives inside that method, not in `fit()`.\n- Step 4 (`_handle_complete_stage_settings_check()`) — yes, in `fit()` this would be something like `if self.workflow_stage == \"complete\": self._handle_complete_stage_settings_check()`. \n\nEach of these steps is just a single method call or a simple conditional check from `fit()`'s perspective — the actual complexity is buried in those delegated methods. So the user's right: my assessment of `fit()`'s complexity is misleading because I was describing the complexity of the helper methods, not the method itself. it's just method calls). The COMPLEX label should apply to the *delegated methods*, not to the step as it appears in `fit()`.\n\nThis is a really good point about the level of abstraction. At the highest-level flow (which is what Phase 1 is about), `fit()` is just orchestrating method calls. The complexity assessment should either:\n1. Reflect complexity from `fit()`'s perspective (all steps are SIMPLE — they're all just method calls)\n2. Reflect complexity of the underlying implementation (which belongs in Phase 2 when we drill into each method)\n\nThe user is correct that at this level of abstraction, from `fit()`'s perspective, these are all single-line method calls. The COMPLEX label was me getting ahead of myself. So from `fit()`'s perspective, all steps are essentially SIMPLE — they're just delegating to other methods, and the real complexity lives in those delegated implementations that we'll explore later."}


**Claude 4.6 Sonnet**
Yes, exactly right. From `fit()`'s perspective, every step is just a single method call:

```python
def fit(self):
    if self.fit_type is None:
        raise ValueError(...)
    
    self._log(f"fit() called. Current stage: {self.workflow_stage}")
    
    self._mark_stale_fits()
    
    if self.workflow_stage == "complete":
        self._handle_complete_stage_settings_check()
    
    self._detect_and_handle_new_locations()
    
    if self.fit_type == 'point lens':
        self.fit_point_lens()
    elif self.fit_type == 'binary lens':
        self.fit_binary_lens()
    
    self._check_renorm_completeness()
    self._output_latex_table()
```

The COMPLEX label in my complexity table was describing the complexity of the *delegated methods themselves*, not of the lines in `fit()`. At this level of abstraction, every step in `fit()` is [SIMPLE] — it's just a method call or a guard condition.

The complexity assessment belongs one level down, when we sketch the pseudocode for each of those delegated methods in later phases.

Does that match what you were pointing out? And does the `fit()` body above look right to you as the complete high-level flow?


**User**
Thanks. 

Second question: Step 4: _handle_complete_stage() 

1. I'm confused by this, because in the one-touch workflows, the workflow_stage isn't complete until after 6.

2. This step has "runs: parallax grid" in it, but I thought that would happen in fit_point_lens().

Am I just confused because we're still missing the lower-level pieces?


**Claude 4.6 Sonnet**
{"type":"think","think":"The user is asking two good questions about Step 4 (`_handle_complete_stage_settings_check()`):\n\n1. In the one-touch workflow, `workflow_stage` won't be \"complete\" when `fit()` is first called — it starts at \"start\". So Step 4 would be skipped entirely on a fresh run. The user is confused because they're thinking about the first run, where `workflow_stage` is never \"complete\" when `fit()` is called. Step 4 only matters when `fit()` is called *after* a previous run has already reached \"complete\" — i.e., on a second call to `fit()`, or after loading a restart file from a completed run.\n\n\n2. For the parallax grid case in Step 4, when `parallax_grid` was False before but is now True and the workflow is already \"complete\", we need to run the grid without resetting everything. But the user's right — if we set the stage appropriately, `fit_point_lens()` would handle it naturally, which circles back to my earlier question about whether Step 4 is even necessary.\n\nLet me reconsider: Step 4 only executes when a previous `fit()` call finished successfully and the user calls `fit()` again, so it's purely for the \"resume after completion\" scenario. On a fresh workflow, `workflow_stage` starts at \"start\" and Step 4's condition would never trigger.\n\nFor the parallax grid specifically, running it inside `_handle_complete_stage_settings_check()` feels awkward. It'd be cleaner to either reset the stage so `fit_point_lens()` runs the grid naturally, or use a flag instead. Looking back at the requirements, when parallax_grid changes to True, we should run the final parallax grid without resetting the stage—this is a special case where we don't want to re-run everything else. If I reset the stage to something like \"pl_fit_done\", the renormalization step would re-examine and potentially re-run, which isn't what we want. The intent is specifically to run just the grid without triggering the full pipeline again.\n\nSo the question becomes whether to handle this directly in `_handle_complete_stage_settings_check()` or use some flag mechanism. The key insight is that `fit_point_lens()` won't even run when `workflow_stage == \"complete\"` since all stages would appear past, so the grid can't execute through the normal flow. Temporarily resetting the stage would violate R2.15's requirement not to reset. This means Step 4 only activates when `workflow_stage` is already \"complete\" from a previous run—in a fresh one-touch workflow, this step wouldn't trigger at all."}


**Claude 4.6 Sonnet**
Both observations are correct. Let me address them separately:

**On question 1:** You're right. In a one-touch workflow, `workflow_stage` starts at `"start"` and Step 4's condition is never true during that run. Step 4 *only* fires when `fit()` is called **after a previous run already reached `"complete"`** — i.e., the user is resuming or re-running after changes. On a fresh run, Step 4 is simply skipped.

**On question 2:** This is actually my Q2 from above, and you're identifying the same tension. The parallax grid running inside `_handle_complete_stage_settings_check()` is awkward for exactly the reason you're sensing — it breaks the pattern of "workflow stages run inside `fit_point_lens()`."

The reason it ended up there is that R2.15 says "run final parallax grid only, **do not reset stage**." If we reset the stage back to something like `"pl_fit_done"`, then `fit_point_lens()` would naturally run the grid — but it would also potentially re-run renormalization, which we don't want.

So: **can you clarify Q2?** Specifically:

- **Option A:** `_handle_complete_stage_settings_check()` calls the grid search method directly (stage stays `"complete"`)
- **Option B:** We reset the stage to a specific point that causes only the grid to run inside `fit_point_lens()`, and rely on skip conditions being precise enough to avoid re-running other steps

Which feels more right to you?


**User**
Sounds like we should work through your questions. Let's do them one at a time, starting with Q1.


**Claude 4.6 Sonnet**
{"type":"think","think":"The user wants to work through my questions one at a time, starting with Q1.\n\nQ1 was:\n\n**Q1 (Settings comparison — highest priority):**\nStep 4 compares current config against a \"previous\" config. My assumption (A1) is that `self._saved_config` holds the config dict loaded from the restart file. But there's a gap: if the user calls `fit()` twice in the same session (no restart file, no state reload between calls), the \"previous config\" would be whatever was in `__init__` — meaning no changes would be detected, which is correct. However, *after* a complete run, the config was saved to the restart file. \n\nOn the next session, when the restart file gets loaded back, I'm comparing the config at `__init__` time against what was persisted from the previous run. I need to clarify whether this cross-session comparison is the intended behavior, or if I should maintain an explicit snapshot that gets updated whenever a stage completes."}


**Claude 4.6 Sonnet**
Great. Here is Q1 restated cleanly:

---

**Q1: What is the "previous config" that Step 4 compares against?**

The settings-change check in R2.15 needs to compare the *current* config against *something*. The natural candidate is the config loaded from the restart file. But I want to confirm this is intentional, because it has a specific implication:

- The comparison is **session-to-session** (current `__init__` args vs. what was saved in the restart file from the previous run)
- It is **not** a within-session comparison (e.g., comparing config at the start of `fit()` against config at the end)

So concretely: if the user loads a restart file, changes `fit_type` from `'point lens'` to `'binary lens'`, and calls `fit()` — the check detects the change because `self.fit_type == 'binary lens'` but `saved_config['fit_type'] == 'point lens'`.

**My question:** Is `saved_config` from the restart file the right thing to compare against, or should there be a dedicated `self._config_at_last_complete` snapshot that gets explicitly saved when `_set_stage("complete")` is called?

The difference matters if a user could reach `"complete"` *within the same session* (no restart file) and then immediately call `fit()` again with changed settings — in that case the restart file might not reflect the just-completed run yet, depending on when `_save_restart_state()` fires.


**User**
1. The comparison is session-to-session . So saved_config from the restart file is the correct thing to compare against.

2. I think that if the user calls fit() twice in a row, nothing should happen. The user should not be allowed to change the settings (e.g. parallax_grid) without creating a new instance of MMEXOFASTFitter.


**Claude 4.6 Sonnet**
{"type":"think","think":"The user is clarifying Q1:\n\n1. The comparison is session-to-session, using `saved_config` from the restart file. Good.\n\n2. The user is saying that settings cannot be changed mid-session - if the user wants to change settings (like `parallax_grid`), they must create a new instance of `MMEXOFASTFitter` with those settings and provide the restart file. So the settings-change detection in R2.15 only happens when loading a restart file with different settings.\n\nThis simplifies things significantly:\n- If there's no restart file, `workflow_stage` starts at `\"start\"` and we never reach `\"complete\"` in `__init__`, so Step 4 never fires on a fresh run.\n\n\n- If there IS a restart file and the stage is `\"complete\"`, we compare current config against `saved_config` from the restart file. But when the user calls `fit()` twice in the same session, the second call finds `workflow_stage == \"complete\"` from the first call, yet there's no restart file to trigger the comparison—so I need to think through whether that's actually a problem or if the logic handles it correctly.\n\nWhen a user creates a fitter with no restart file and calls `fit()` twice back-to-back, the second call should be a no-op. The `saved_config` stays empty since no restart file existed at initialization, so the settings-change check finds nothing different, and combined with no stale fits or new locations, the entire second call becomes a no-op—which is the desired behavior.\n\nThe user's clarification reveals that settings changes are intentional and require both a new fitter instance with updated settings AND the restart file provided. This means `saved_config` from the restart file is the right comparison target, eliminating the need for a separate `_config_at_last_complete` attribute. For the `parallax_grid` scenario, when a user creates a new fitter with different settings and provides the restart file, the comparison will detect the change. So the parallax grid needs to execute in the final workflow step when it's been toggled to True, and I should move on to addressing Q2 now."}


**Claude 4.6 Sonnet**
That simplifies things significantly. So the full picture for Q1 is:

- Settings changes only happen via a new `MMEXOFASTFitter` instance, which means a restart file must be provided for the new instance to inherit state from the previous run
- `saved_config` in `__init__` comes from that restart file
- If no restart file was provided, `saved_config = {}`, so no settings changes are detected — and calling `fit()` twice in a row is naturally a no-op (stage is `"complete"`, no changes detected, no stale fits)
- The R2.15 settings-change scenarios all require: previous run reached `"complete"` and was saved, plus a new instance created with different settings pointing at that restart file

This means `saved_config` from `_load_restart_data()` is exactly the right comparison target, and no dedicated `_config_at_last_complete` attribute is needed.

---

Shall we move to **Q2** now?

To restate it cleanly in light of your answer to Q1:

**Q2: Where does the "run final parallax grid only" scenario from R2.15 execute?**

The user creates a new instance with `parallax_grid=True`, providing the restart file from a completed run. `workflow_stage == "complete"`, and Step 4 detects `parallax_grid` changed `False → True`. 

Does the grid run inside `_handle_complete_stage_settings_check()` directly, or does Step 4 do something that causes it to run inside `fit_point_lens()`?


