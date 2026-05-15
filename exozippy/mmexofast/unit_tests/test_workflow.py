# test_workflow.py

import pickle
import os
import glob
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import exozippy
import exozippy.mmexofast as mmexo
from exozippy.mmexofast import MMEXOFASTFitter, WorkflowStep
from exozippy.mmexofast.fitters import MulensFitter

# OB05390
OB05390_FILES = sorted(glob.glob(os.path.join(
    exozippy.MULENS_DATA_PATH, 'OB05390', 'n200*.txt')))

with open(os.path.join(
        exozippy.MULENS_DATA_PATH, 'OB05390', 'coords.txt')) as f:
    OB05390_COORDS = f.read().strip()

BINARY_FIT_KEY = mmexo.FitKey(
    lens_type=mmexo.LensType.BINARY,
    source_type=mmexo.SourceType.FINITE,
    parallax_branch=mmexo.ParallaxBranch.NONE,
    lens_orb_motion=mmexo.LensOrbMotion.NONE,
    locations_used=None,
)

BINARY_PARAMS = {
    't_0':   2453582.7281740606,
    'u_0':   0.355227507989543,
    't_E':   11.106795114521415,
    'rho':   0.024632765186197645,
    'q':     7.524529162733864e-05,
    's':     1.6044784697939465,
    'alpha': 157.9506556145345,
}

BEST_EF_GRID_POINT = {
    't_0':   2456836.080383359,
    't_eff': 23.67696884508345,
    'j':     2,
    'chi2':  -137842.8089725696,
}

# OB140939
GROUND_DATA_FILES = [os.path.join(
    exozippy.MULENS_DATA_PATH, 'OB140939',
    'n20100310.I.OGLE.OB140939.txt')]

COORDS = '17:47:12.25 -21:22:58.7'

STATIC_PSPL_KEY = mmexo.FitKey(
    lens_type=mmexo.LensType.POINT,
    source_type=mmexo.SourceType.POINT,
    parallax_branch=mmexo.ParallaxBranch.NONE,
    lens_orb_motion=mmexo.LensOrbMotion.NONE,
    locations_used=None,
)

STATIC_PSPL_PARAMS = {
    't_0': 2456836.,
    'u_0': 1.012,
    't_E': 21.48,
}

INITIAL_RESULTS = {
    'PSPL static': {
        'params': STATIC_PSPL_PARAMS,
    }
}

# ---------------------------------------------------------------------------
# Per-stage step blocks
#
# Each stage's steps are defined once. The combined workflow lists below are
# built by concatenation, so adding or renaming a step in one place
# propagates everywhere automatically.
# ---------------------------------------------------------------------------

_STEPS_EVENT_SEARCH = [
    ('run_ef_grid', 'event_search'),
]
_STEPS_STATIC_PL = [
    ('est_pl_params', 'fit_static_point_lens'),
    ('fit_pspl',      'fit_static_point_lens'),
]
_STEPS_PL_PARALLAX = [
    ('fit_parallax_u0+', 'fit_point_lens_parallax'),
    ('fit_parallax_u0-', 'fit_point_lens_parallax'),
]
_STEPS_RENORM = [
    ('renormalize_datasets', 'renormalize'),
    ('refit_all',            'renormalize'),
]
_STEPS_SEARCH_ANOMALY = [
    ('select_best_point_lens_model', 'search_for_anomaly'),
    ('compute_residuals',            'search_for_anomaly'),
    ('run_af_grid',                  'search_for_anomaly'),
    ('get_anomaly_lc_params',        'search_for_anomaly'),
    ('classify_anomaly',             'search_for_anomaly'),
]
_STEPS_FIT_BINARY = [
    ('est_binary_params', 'fit_binary_lens'),
    ('fit_binary_models', 'fit_binary_lens'),
]
_STEPS_CHECK_BINARY_RENORM = [
    ('check_needs_renorm', 'check_binary_renorm'),
]
_STEPS_PARALLAX_GRIDS = [
    ('run_parallax_grids', 'parallax_grids'),
]

# Combined workflow lists, each built entirely from the blocks above.
EXPECTED_STEPS = (
    _STEPS_EVENT_SEARCH +
    _STEPS_STATIC_PL +
    _STEPS_PL_PARALLAX
)
EXPECTED_STEPS_PL_RENORM = (
    _STEPS_EVENT_SEARCH +
    _STEPS_STATIC_PL +
    _STEPS_PL_PARALLAX +
    _STEPS_RENORM +
    _STEPS_PARALLAX_GRIDS
)
EXPECTED_STEPS_BINARY = (
    _STEPS_EVENT_SEARCH +
    _STEPS_STATIC_PL +
    _STEPS_PL_PARALLAX +
    _STEPS_RENORM +
    _STEPS_SEARCH_ANOMALY +
    _STEPS_FIT_BINARY +
    _STEPS_CHECK_BINARY_RENORM +
    _STEPS_PARALLAX_GRIDS
)

# ---------------------------------------------------------------------------
# Name-based slice helper
#
# Replaces magic-index slicing (e.g. EXPECTED_STEPS_BINARY[:11]) with an
# intent-revealing name. A wrong index silently tests the wrong scenario;
# a wrong name raises ValueError immediately.
# ---------------------------------------------------------------------------

def steps_through(steps, step_name):
    """
    Return a prefix of `steps` ending at (and including) the first step
    whose name matches `step_name`.

    Parameters
    ----------
    steps : list of (name, stage) tuples
    step_name : str

    Returns
    -------
    list of (name, stage) tuples

    Raises
    ------
    ValueError
        If step_name is not present in steps.
    """
    for i, (name, _stage) in enumerate(steps):
        if name == step_name:
            return steps[:i + 1]
    raise ValueError(f'{step_name!r} not found in steps list')


# ---------------------------------------------------------------------------
# Step-to-method mapping and shared patching helper
#
# Maps each workflow step name to the MMEXOFASTFitter method it invokes.
# fit_parallax_u0+ and fit_parallax_u0- share one underlying method so one
# patch covers both.
#
# When a new step is added to any EXPECTED_STEPS_* constant, add its entry
# here too. patch_fitter_methods will raise AssertionError immediately if a
# step name is missing, making the omission obvious.
# ---------------------------------------------------------------------------

_STEP_TO_METHOD = {
    'run_ef_grid':                  'run_ef_grid',
    'est_pl_params':                'est_pl_params',
    'fit_pspl':                     'fit_pspl',
    'fit_parallax_u0+':             'fit_parallax',   # shared method
    'fit_parallax_u0-':             'fit_parallax',   # shared method
    'renormalize_datasets':         'renormalize_datasets',
    'refit_all':                    'refit_all',
    'select_best_point_lens_model': 'select_best_point_lens_model',
    'compute_residuals':            'compute_residuals',
    'run_af_grid':                  'run_af_grid',
    'get_anomaly_lc_params':        'get_anomaly_lc_params',
    'classify_anomaly':             'classify_anomaly',
    'est_binary_params':            'est_binary_params',
    'fit_binary_models':            'fit_binary_models',
    'check_needs_renorm':           'check_needs_renorm',
    'run_parallax_grids':           'run_parallax_grids',
}

# Methods whose no-op return value must be something other than None.
_METHOD_RETURN_VALUES = {
    'run_ef_grid':   {},
    'est_pl_params': {},
}


def patch_fitter_methods(test_case, fitter, expected_steps):
    """
    Patch the fitter methods invoked by expected_steps with no-ops.

    Each step name is looked up in _STEP_TO_METHOD to find the underlying
    fitter method. Steps that share a method (fit_parallax_u0+/u0-) produce
    one patch whose mock is accessible under both step-name keys.

    Parameters
    ----------
    test_case : unittest.TestCase
        Used for the coverage guard assertion.
    fitter : MMEXOFASTFitter
        Instance whose methods are patched.
    expected_steps : list of (name, stage) tuples
        Full workflow for this test class; determines which methods are
        patched and what the guard checks against.

    Returns
    -------
    ExitStack
        Active context manager. Attribute .mocks is a dict keyed by step
        name, mapping to the corresponding MagicMock.

    Raises
    ------
    AssertionError
        If any step name in expected_steps has no entry in _STEP_TO_METHOD.
        Add the missing entry to _STEP_TO_METHOD to resolve.
    """
    # Guard first — fail clearly before patching anything.
    # If this fires, a step was added to an EXPECTED_STEPS_* constant
    # without a corresponding entry in _STEP_TO_METHOD.
    unknown = {name for name, _ in expected_steps} - set(_STEP_TO_METHOD)
    test_case.assertFalse(
        unknown,
        f'Steps missing from _STEP_TO_METHOD: {unknown}. '
        f'Add an entry to _STEP_TO_METHOD when adding a new workflow step.')

    stack = ExitStack()
    method_mocks = {}   # keyed by method name; prevents double-patching
    mocks = {}          # keyed by step name for caller inspection

    for step_name, _ in expected_steps:
        method_name = _STEP_TO_METHOD[step_name]
        if method_name not in method_mocks:
            rv = _METHOD_RETURN_VALUES.get(method_name, None)
            method_mocks[method_name] = stack.enter_context(
                patch.object(fitter, method_name, return_value=rv))
        mocks[step_name] = method_mocks[method_name]

    stack.mocks = mocks
    return stack


def _make_noop_steps(expected_steps):
    """
    Build a WorkflowStep list from (name, stage) tuples with no-op actions.
    Used to pre-populate completed_steps in resume tests.
    """
    return [
        WorkflowStep(
            name=name,
            stage=stage,
            func=MagicMock(return_value=None),
            description=f'No-op for {name}',
        )
        for name, stage in expected_steps
    ]


class TestPointLensWorkflow(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type='point_lens',
            renormalize_errors=False)
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        return patch_fitter_methods(self, fitter, EXPECTED_STEPS)

    # --- dry run ---

    def test_dry_run_planned_steps(self):
        """
        Dry run for ground-only point-lens fit with renormalize_errors=False
        produces the expected step queue with correct names and stages.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.fit()

        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, EXPECTED_STEPS)

    # --- stop_before stage:step ---

    def test_stop_before_first_step_of_stage(self):
        """
        stop_before='fit_static_point_lens:est_pl_params' executes only
        steps before est_pl_params (i.e. the event_search stage only).
        """
        fitter = self._make_fitter(
            stop_before='fit_static_point_lens:est_pl_params')

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = _STEPS_EVENT_SEARCH
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_stop_before_second_step_of_stage(self):
        """
        stop_before='fit_static_point_lens:fit_pspl' executes steps through
        est_pl_params but not fit_pspl.
        """
        fitter = self._make_fitter(
            stop_before='fit_static_point_lens:fit_pspl')

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(EXPECTED_STEPS, 'est_pl_params')
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    # --- stop_after stage:step ---

    def test_stop_after_first_step_of_stage(self):
        """
        stop_after='fit_static_point_lens:est_pl_params' executes steps
        through and including est_pl_params.
        """
        fitter = self._make_fitter(
            stop_after='fit_static_point_lens:est_pl_params')

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(EXPECTED_STEPS, 'est_pl_params')
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_stop_after_second_step_of_stage(self):
        """
        stop_after='fit_static_point_lens:fit_pspl' executes steps through
        and including fit_pspl.
        """
        fitter = self._make_fitter(
            stop_after='fit_static_point_lens:fit_pspl')

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(EXPECTED_STEPS, 'fit_pspl')
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    # --- stop_before stage-only ---

    def test_stop_before_stage_halts_before_first_step(self):
        """
        stop_before='fit_static_point_lens' halts before the first step
        of that stage.
        """
        fitter = self._make_fitter(stop_before='fit_static_point_lens')

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = _STEPS_EVENT_SEARCH
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    # --- stop_after stage-only ---

    def test_stop_after_stage_halts_after_last_step(self):
        """
        stop_after='fit_static_point_lens' halts after the last step
        of that stage (i.e. after fit_pspl).
        """
        fitter = self._make_fitter(stop_after='fit_static_point_lens')

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(EXPECTED_STEPS, 'fit_pspl')
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    # --- resume after stop ---

    def test_resume_after_stop_before_planned_steps(self):
        """
        When completed_steps contains only run_ef_grid, planned_steps
        starts at est_pl_params.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(
            steps_through(EXPECTED_STEPS, 'run_ef_grid'))
        fitter.fit()

        expected = _STEPS_STATIC_PL + _STEPS_PL_PARALLAX
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_resume_after_stop_after_planned_steps(self):
        """
        When completed_steps contains run_ef_grid and est_pl_params,
        planned_steps starts at fit_pspl.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(
            steps_through(EXPECTED_STEPS, 'est_pl_params'))
        fitter.fit()

        expected = _STEPS_STATIC_PL[1:] + _STEPS_PL_PARALLAX
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)


class TestPointLensRenormWorkflow(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type='point_lens',
            renormalize_errors=True,
            parallax_grid=True)
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        return patch_fitter_methods(self, fitter, EXPECTED_STEPS_PL_RENORM)

    def test_dry_run_planned_steps(self):
        """
        Dry run for ground-only point-lens fit with renormalize_errors=True
        and parallax_grid=True produces the expected step queue.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.fit()

        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, EXPECTED_STEPS_PL_RENORM)

    def test_stop_before_renormalize_stage(self):
        """
        stop_before='renormalize' halts before renormalize_datasets.
        """
        fitter = self._make_fitter(stop_before='renormalize')

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(EXPECTED_STEPS_PL_RENORM, 'fit_parallax_u0-')
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_stop_after_renormalize_stage(self):
        """
        stop_after='renormalize' halts after refit_all.
        """
        fitter = self._make_fitter(stop_after='renormalize')

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(EXPECTED_STEPS_PL_RENORM, 'refit_all')
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_resume_after_stop_before_renormalize(self):
        """
        When completed_steps ends at fit_point_lens_parallax,
        planned_steps starts at renormalize_datasets.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(
            steps_through(EXPECTED_STEPS_PL_RENORM, 'fit_parallax_u0-'))
        fitter.fit()

        expected = _STEPS_RENORM + _STEPS_PARALLAX_GRIDS
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)


class TestBinaryLensWorkflow(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type='binary_lens',
            renormalize_errors=True,
            parallax_grid=True)
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        return patch_fitter_methods(self, fitter, EXPECTED_STEPS_BINARY)

    def test_dry_run_planned_steps(self):
        """
        Dry run for ground-only binary lens fit with renormalize_errors=True
        and parallax_grid=True produces the expected step queue.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.fit()

        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, EXPECTED_STEPS_BINARY)

    def test_stop_before_search_for_anomaly(self):
        """
        stop_before='search_for_anomaly' halts before
        select_best_point_lens_model.
        """
        fitter = self._make_fitter(stop_before='search_for_anomaly')

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(EXPECTED_STEPS_BINARY, 'refit_all')
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_stop_after_fit_binary(self):
        """
        stop_after='fit_binary_lens' halts after fit_binary_models.
        """
        fitter = self._make_fitter(stop_after='fit_binary_lens')

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(EXPECTED_STEPS_BINARY, 'fit_binary_models')
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_resume_after_stop_before_fit_binary(self):
        """
        When completed_steps ends at est_binary_params,
        planned_steps starts at fit_binary_models.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(
            steps_through(EXPECTED_STEPS_BINARY, 'est_binary_params'))
        fitter.fit()

        expected = (
            _STEPS_FIT_BINARY[1:] +
            _STEPS_CHECK_BINARY_RENORM +
            _STEPS_PARALLAX_GRIDS
        )
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_resume_after_stop_after_check_binary_renorm(self):
        """
        When completed_steps ends at check_binary_renorm,
        planned_steps contains only the parallax grid steps.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(
            steps_through(EXPECTED_STEPS_BINARY, 'check_needs_renorm'))
        fitter.fit()

        expected = _STEPS_PARALLAX_GRIDS
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_check_needs_renorm_inserts_steps_when_true(self):
        """
        When check_needs_renorm returns True, renormalize_datasets and
        refit_all are inserted and executed before run_parallax_grids.
        """
        fitter = self._make_fitter()
        fitter.completed_steps = _make_noop_steps(
            steps_through(EXPECTED_STEPS_BINARY, 'fit_binary_models'))

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(fitter, '_needs_renormalization',
                             return_value=True))
            stack.enter_context(
                patch.object(fitter, 'renormalize_datasets',
                             return_value=None))
            stack.enter_context(
                patch.object(fitter, 'refit_all',
                             return_value=None))
            stack.enter_context(
                patch.object(fitter, 'run_parallax_grids',
                             return_value=None))
            fitter.fit()

        actual_names = [step.name for step in fitter.completed_steps]
        self.assertIn('renormalize_datasets', actual_names)
        self.assertIn('refit_all', actual_names)

        renorm_idx = actual_names.index('renormalize_datasets')
        grid_idx = actual_names.index('run_parallax_grids')
        self.assertLess(renorm_idx, grid_idx)


class TestPointLensWorkflowWithInitialResults(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type='point_lens',
            renormalize_errors=False,
            initial_results=INITIAL_RESULTS)
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        # Patch the full point-lens workflow; the initial_results causes
        # some steps to be skipped at runtime, but patching unused methods
        # is harmless and keeps the guard meaningful.
        return patch_fitter_methods(self, fitter, EXPECTED_STEPS)

    def test_dry_run_skips_est_pl_params(self):
        """
        When a static PSPL is supplied via initial_results with
        fit_type='point_lens', planned_steps starts at fit_pspl,
        skipping est_pl_params.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.fit()

        expected = _STEPS_STATIC_PL[1:] + _STEPS_PL_PARALLAX
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_fit_pspl_uses_supplied_params_as_seed(self):
        """
        fit_pspl is called with the user-supplied PSPL params as seed.
        """
        fitter = self._make_fitter()

        with self._patch_fit_methods(fitter) as stack:
            fitter.fit()

        call_args = stack.mocks['fit_pspl'].call_args
        self.assertEqual(
            call_args.kwargs.get('initial_params'), STATIC_PSPL_PARAMS)

    def test_initial_results_and_restart_from_raises(self):
        """
        Providing both initial_results and restart_from raises ValueError.
        """
        with self.assertRaises(ValueError):
            self._make_fitter(restart_from='event_search')


class TestBinaryLensWorkflowWithInitialResults(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type='binary_lens',
            renormalize_errors=False,
            initial_results=INITIAL_RESULTS)
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        return patch_fitter_methods(
            self, fitter, _STEPS_SEARCH_ANOMALY + _STEPS_FIT_BINARY)

    def test_dry_run_starts_at_search_for_anomaly(self):
        """
        When a static PSPL is supplied via initial_results with
        fit_type='binary_lens', planned_steps starts at search_for_anomaly,
        skipping all point-lens stages.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.fit()

        expected = _STEPS_SEARCH_ANOMALY + _STEPS_FIT_BINARY
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_select_best_point_lens_model_returns_supplied_pspl(self):
        """
        select_best_point_lens_model returns the user-supplied PSPL record
        when initial_results contains a PSPL fit.
        """
        fitter = self._make_fitter()
        result = fitter.select_best_point_lens_model()
        self.assertEqual(result.params, STATIC_PSPL_PARAMS)

    def test_initial_results_and_restart_from_raises(self):
        """
        Providing both initial_results and restart_from raises ValueError.
        """
        with self.assertRaises(ValueError):
            self._make_fitter(restart_from='event_search')


def _make_fake_pickle(path, completed_steps):
    """
    Write a minimal fake restart pickle containing only completed_steps.
    """
    state = {
        'completed_steps': [(s.name, s.stage) for s in completed_steps],
    }
    data = {'config': {}, 'state': state}
    with open(path, 'wb') as f:
        pickle.dump(data, f)


class TestBinaryLensRestartFromPointLens(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, restart_file, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type='binary_lens',
            renormalize_errors=True)
        defaults.update(kwargs)
        return MMEXOFASTFitter(restart_file=restart_file, **defaults)

    def test_binary_steps_added_after_point_lens_restart(self):
        """
        Restarting from a completed point-lens run (renormalize_errors=True,
        parallax_grid=False) with fit_type='binary_lens' produces a step
        queue that starts at search_for_anomaly.
        """
        pl_completed = _make_noop_steps(
            steps_through(EXPECTED_STEPS_BINARY, 'refit_all'))

        pkl_path = os.path.join(self.tmp_path, 'fake.pkl')
        _make_fake_pickle(pkl_path, pl_completed)

        fitter = self._make_fitter(restart_file=pkl_path, dry_run=True)
        fitter.fit()

        expected = (
            _STEPS_SEARCH_ANOMALY +
            _STEPS_FIT_BINARY +
            _STEPS_CHECK_BINARY_RENORM
        )
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)


class TestExecutionLoopDynamicSteps(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_action_returning_none_does_not_insert_steps(self):
        """
        est_pl_params runs for real, returns None from the step action,
        and the execution loop continues normally without inserting steps.
        The result is stored in intermediate_results.est_pl_params.
        """
        fitter = MMEXOFASTFitter(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type='point_lens',
            renormalize_errors=False,
            stop_after='fit_static_point_lens:est_pl_params')

        fitter.completed_steps = _make_noop_steps(_STEPS_EVENT_SEARCH)
        fitter.intermediate_results.best_ef_grid_point = BEST_EF_GRID_POINT

        fitter.fit()

        self.assertIsNotNone(fitter.intermediate_results.est_pl_params)
        self.assertEqual(fitter.completed_steps[-1].name, 'est_pl_params')

    def test_action_returning_steps_inserts_at_front_of_queue(self):
        """
        check_needs_renorm runs for real with a binary FitRecord that causes
        _needs_renormalization() to return True. The dynamic renorm steps are
        inserted at the front of the queue and visible in planned_steps before
        they execute.
        """
        fitter = MMEXOFASTFitter(
            files=OB05390_FILES,
            coords=OB05390_COORDS,
            fit_type='binary_lens',
            renormalize_errors=True,
            parallax_grid=True,
            stop_after='check_binary_renorm:check_needs_renorm')

        fitter.completed_steps = _make_noop_steps(
            steps_through(EXPECTED_STEPS_BINARY, 'fit_binary_models'))

        binary_fitter = MulensFitter(
            datasets=fitter.datasets,
            initial_model_params=BINARY_PARAMS,
            mag_methods=[2453591., 'VBBL', 2453594.],
            coords=OB05390_COORDS)
        binary_fitter.best = binary_fitter.initial_model_params
        binary_fitter.best['chi2'] = 562.

        binary_record = mmexo.FitRecord.from_full_result(
            model_key=BINARY_FIT_KEY,
            full_result=mmexo.MMEXOFASTFitResults(binary_fitter))

        fitter.all_fit_results.set(binary_record)
        fitter.fit()

        completed_names = [(step.name, step.stage)
                           for step in fitter.completed_steps]
        self.assertIn(('check_needs_renorm', 'check_binary_renorm'),
                      completed_names)

        planned_names = [(step.name, step.stage)
                         for step in fitter.planned_steps]
        self.assertIn(('renormalize_datasets', 'check_binary_renorm'),
                      planned_names)
        self.assertIn(('refit_all', 'check_binary_renorm'),
                      planned_names)

        names_only = [name for name, _ in planned_names]
        self.assertLess(
            names_only.index('renormalize_datasets'),
            names_only.index('run_parallax_grids'))


class TestSelectBestPointLensModel(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type='point lens',
            renormalize_errors=False)
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _make_static_key(self, locations_used=None):
        return mmexo.FitKey(
            lens_type=mmexo.LensType.POINT,
            source_type=mmexo.SourceType.POINT,
            parallax_branch=mmexo.ParallaxBranch.NONE,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
            locations_used=locations_used,
        )

    def _make_parallax_key(self, branch=mmexo.ParallaxBranch.U0_PLUS):
        return mmexo.FitKey(
            lens_type=mmexo.LensType.POINT,
            source_type=mmexo.SourceType.POINT,
            parallax_branch=branch,
            lens_orb_motion=mmexo.LensOrbMotion.NONE,
        )

    def _make_record(self, key, chi2_value=None):
        record = mmexo.FitRecord(
            model_key=key,
            params=STATIC_PSPL_PARAMS,
            is_complete=(chi2_value is not None),
        )
        if chi2_value is not None:
            record.chi2 = lambda: chi2_value
        return record

    def test_raises_when_no_point_lens_fits(self):
        fitter = self._make_fitter()
        with self.assertRaises(RuntimeError):
            fitter.select_best_point_lens_model()

    def test_multiple_incomplete_records_raises(self):
        fitter = self._make_fitter()
        fitter.all_fit_results.set(self._make_record(self._make_static_key()))
        fitter.all_fit_results.set(self._make_record(self._make_parallax_key()))
        with self.assertRaises(RuntimeError):
            fitter.select_best_point_lens_model()

    def test_static_fits_only_returns_best_chi2(self):
        fitter = self._make_fitter()
        better = self._make_record(
            self._make_static_key(locations_used='a'), chi2_value=100.0)
        worse = self._make_record(
            self._make_static_key(locations_used='b'), chi2_value=200.0)
        fitter.all_fit_results.set(better)
        fitter.all_fit_results.set(worse)
        self.assertIs(fitter.select_best_point_lens_model(), better)

    def test_parallax_fits_only_returns_best_chi2(self):
        fitter = self._make_fitter()
        better = self._make_record(
            self._make_parallax_key(mmexo.ParallaxBranch.U0_PLUS),
            chi2_value=80.0)
        worse = self._make_record(
            self._make_parallax_key(mmexo.ParallaxBranch.U0_MINUS),
            chi2_value=120.0)
        fitter.all_fit_results.set(better)
        fitter.all_fit_results.set(worse)
        self.assertIs(fitter.select_best_point_lens_model(), better)

    def test_returns_static_when_parallax_improvement_below_threshold(self):
        fitter = self._make_fitter()
        static = self._make_record(
            self._make_static_key(), chi2_value=1000.0)
        parallax = self._make_record(
            self._make_parallax_key(), chi2_value=960.0)  # improvement = 40
        fitter.all_fit_results.set(static)
        fitter.all_fit_results.set(parallax)
        self.assertIs(fitter.select_best_point_lens_model(), static)

    def test_returns_parallax_when_improvement_above_threshold(self):
        fitter = self._make_fitter()
        static = self._make_record(
            self._make_static_key(), chi2_value=1000.0)
        parallax = self._make_record(
            self._make_parallax_key(), chi2_value=900.0)  # improvement = 100
        fitter.all_fit_results.set(static)
        fitter.all_fit_results.set(parallax)
        self.assertIs(fitter.select_best_point_lens_model(), parallax)

    def test_incomplete_records_ignored_when_complete_records_exist(self):
        fitter = self._make_fitter()
        complete_static = self._make_record(
            self._make_static_key(), chi2_value=1000.0)
        incomplete_parallax = self._make_record(
            self._make_parallax_key(), chi2_value=None)
        fitter.all_fit_results.set(complete_static)
        fitter.all_fit_results.set(incomplete_parallax)
        self.assertIs(fitter.select_best_point_lens_model(), complete_static)
