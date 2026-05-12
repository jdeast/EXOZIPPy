# test_workflow.py


import pickle
import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import exozippy
import exozippy.mmexofast as mmexo
from exozippy.mmexofast import MMEXOFASTFitter, WorkflowStep

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
    STATIC_PSPL_KEY: {
        'params': STATIC_PSPL_PARAMS,
    }
}

# (name, stage)
EXPECTED_STEPS = [
    ('run_ef_grid',           'event_search'),
    ('est_pl_params',         'fit_static_point_lens'),
    ('fit_pspl',              'fit_static_point_lens'),
    ('fit_parallax_u0_plus',  'fit_point_lens_parallax'),
    ('fit_parallax_u0_minus', 'fit_point_lens_parallax'),
]

EXPECTED_STEPS_PL_RENORM = [
    ('run_ef_grid',                'event_search'),
    ('est_pl_params',              'fit_static_point_lens'),
    ('fit_pspl',                   'fit_static_point_lens'),
    ('fit_parallax_u0_plus',       'fit_point_lens_parallax'),
    ('fit_parallax_u0_minus',      'fit_point_lens_parallax'),
    ('renormalize_datasets',       'renormalize'),
    ('refit_all',                  'renormalize'),
    ('run_parallax_grid_u0_plus',  'run_parallax_grids'),
    ('run_parallax_grid_u0_minus', 'run_parallax_grids'),
]

EXPECTED_STEPS_BINARY = [
    ('run_ef_grid',                  'event_search'),
    ('est_pl_params',                'fit_static_point_lens'),
    ('fit_pspl',                     'fit_static_point_lens'),
    ('fit_parallax_u0_plus',         'fit_point_lens_parallax'),
    ('fit_parallax_u0_minus',        'fit_point_lens_parallax'),
    ('renormalize_datasets',         'renormalize'),
    ('refit_all',                    'renormalize'),
    ('select_best_point_lens_model', 'search_for_anomaly'),
    ('compute_residuals',            'search_for_anomaly'),
    ('run_af_grid',                  'search_for_anomaly'),
    ('est_binary_params',            'search_for_anomaly'),
    ('fit_binary_models',            'fit_binary'),
    ('check_needs_renorm',           'check_binary_renorm'),
    ('run_parallax_grid_u0_plus',    'run_parallax_grids'),
    ('run_parallax_grid_u0_minus',   'run_parallax_grids'),
]


def _make_noop_steps(expected_steps):
    """
    Build a WorkflowStep list from (name, stage) tuples with no-op actions.
    Used to pre-populate completed_steps in resume tests.
    """
    return [
        WorkflowStep(
            name=name,
            stage=stage,
            action=MagicMock(return_value=None),
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
            fit_type='point lens',
            renormalize_errors=False)
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        """
        Patch all fit action methods with no-ops so the execution loop
        runs without performing real fits. Returns an ExitStack with a
        .mocks dict attribute for post-call inspection.
        """
        stack = ExitStack()
        mocks = {}
        mocks['run_ef_grid'] = stack.enter_context(
            patch.object(fitter, 'run_ef_grid', return_value={}))
        mocks['est_pl_params'] = stack.enter_context(
            patch.object(fitter, 'est_pl_params', return_value={}))
        mocks['fit_pspl'] = stack.enter_context(
            patch.object(fitter, 'fit_pspl', return_value=None))
        mocks['fit_parallax'] = stack.enter_context(
            patch.object(fitter, 'fit_parallax', return_value=None))
        stack.mocks = mocks
        return stack

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

        expected = [('run_ef_grid', 'event_search')]
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

        expected = [
            ('run_ef_grid',   'event_search'),
            ('est_pl_params', 'fit_static_point_lens'),
        ]
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

        expected = [
            ('run_ef_grid',   'event_search'),
            ('est_pl_params', 'fit_static_point_lens'),
        ]
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

        expected = [
            ('run_ef_grid',   'event_search'),
            ('est_pl_params', 'fit_static_point_lens'),
            ('fit_pspl',      'fit_static_point_lens'),
        ]
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

        expected = [('run_ef_grid', 'event_search')]
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

        expected = [
            ('run_ef_grid',   'event_search'),
            ('est_pl_params', 'fit_static_point_lens'),
            ('fit_pspl',      'fit_static_point_lens'),
        ]
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    # --- resume after stop ---

    def test_resume_after_stop_before_planned_steps(self):
        """
        When completed_steps contains only run_ef_grid, planned_steps
        starts at est_pl_params.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(EXPECTED_STEPS[:1])
        fitter.fit()

        expected = [
            ('est_pl_params',         'fit_static_point_lens'),
            ('fit_pspl',              'fit_static_point_lens'),
            ('fit_parallax_u0_plus',  'fit_point_lens_parallax'),
            ('fit_parallax_u0_minus', 'fit_point_lens_parallax'),
        ]
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_resume_after_stop_after_planned_steps(self):
        """
        When completed_steps contains run_ef_grid and est_pl_params,
        planned_steps starts at fit_pspl.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(EXPECTED_STEPS[:2])
        fitter.fit()

        expected = [
            ('fit_pspl',              'fit_static_point_lens'),
            ('fit_parallax_u0_plus',  'fit_point_lens_parallax'),
            ('fit_parallax_u0_minus', 'fit_point_lens_parallax'),
        ]
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
            fit_type='point lens',
            renormalize_errors=True,
            parallax_grid=True)
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        stack = ExitStack()
        mocks = {}
        mocks['run_ef_grid'] = stack.enter_context(
            patch.object(fitter, 'run_ef_grid', return_value={}))
        mocks['est_pl_params'] = stack.enter_context(
            patch.object(fitter, 'est_pl_params', return_value={}))
        mocks['fit_pspl'] = stack.enter_context(
            patch.object(fitter, 'fit_pspl', return_value=None))
        mocks['fit_parallax'] = stack.enter_context(
            patch.object(fitter, 'fit_parallax', return_value=None))
        mocks['renormalize_datasets'] = stack.enter_context(
            patch.object(fitter, 'renormalize_datasets', return_value=None))
        mocks['refit_all'] = stack.enter_context(
            patch.object(fitter, 'refit_all', return_value=None))
        mocks['run_parallax_grid'] = stack.enter_context(
            patch.object(fitter, 'run_parallax_grid', return_value=None))
        stack.mocks = mocks
        return stack

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

        expected = [
            ('run_ef_grid',           'event_search'),
            ('est_pl_params',         'fit_static_point_lens'),
            ('fit_pspl',              'fit_static_point_lens'),
            ('fit_parallax_u0_plus',  'fit_point_lens_parallax'),
            ('fit_parallax_u0_minus', 'fit_point_lens_parallax'),
        ]
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_stop_after_renormalize_stage(self):
        """
        stop_after='renormalize' halts after refit_all.
        """
        fitter = self._make_fitter(stop_after='renormalize')

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = [
            ('run_ef_grid',           'event_search'),
            ('est_pl_params',         'fit_static_point_lens'),
            ('fit_pspl',              'fit_static_point_lens'),
            ('fit_parallax_u0_plus',  'fit_point_lens_parallax'),
            ('fit_parallax_u0_minus', 'fit_point_lens_parallax'),
            ('renormalize_datasets',  'renormalize'),
            ('refit_all',             'renormalize'),
        ]
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_resume_after_stop_before_renormalize(self):
        """
        When completed_steps ends at fit_point_lens_parallax,
        planned_steps starts at renormalize_datasets.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(EXPECTED_STEPS_PL_RENORM[:5])
        fitter.fit()

        expected = [
            ('renormalize_datasets',       'renormalize'),
            ('refit_all',                  'renormalize'),
            ('run_parallax_grid_u0_plus',  'run_parallax_grids'),
            ('run_parallax_grid_u0_minus', 'run_parallax_grids'),
        ]
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
            fit_type='binary lens',
            renormalize_errors=True,
            parallax_grid=True)
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        stack = ExitStack()
        mocks = {}
        mocks['run_ef_grid'] = stack.enter_context(
            patch.object(fitter, 'run_ef_grid', return_value={}))
        mocks['est_pl_params'] = stack.enter_context(
            patch.object(fitter, 'est_pl_params', return_value={}))
        mocks['fit_pspl'] = stack.enter_context(
            patch.object(fitter, 'fit_pspl', return_value=None))
        mocks['fit_parallax'] = stack.enter_context(
            patch.object(fitter, 'fit_parallax', return_value=None))
        mocks['renormalize_datasets'] = stack.enter_context(
            patch.object(fitter, 'renormalize_datasets', return_value=None))
        mocks['refit_all'] = stack.enter_context(
            patch.object(fitter, 'refit_all', return_value=None))
        mocks['select_best_point_lens_model'] = stack.enter_context(
            patch.object(fitter, 'select_best_point_lens_model',
                         return_value=None))
        mocks['compute_residuals'] = stack.enter_context(
            patch.object(fitter, 'compute_residuals', return_value=None))
        mocks['run_af_grid'] = stack.enter_context(
            patch.object(fitter, 'run_af_grid', return_value=None))
        mocks['est_binary_params'] = stack.enter_context(
            patch.object(fitter, 'est_binary_params', return_value=None))
        mocks['fit_binary_models'] = stack.enter_context(
            patch.object(fitter, 'fit_binary_models', return_value=None))
        mocks['check_needs_renorm'] = stack.enter_context(
            patch.object(fitter, 'check_needs_renorm', return_value=None))
        mocks['run_parallax_grid'] = stack.enter_context(
            patch.object(fitter, 'run_parallax_grid', return_value=None))
        stack.mocks = mocks
        return stack

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

        expected = [
            ('run_ef_grid',           'event_search'),
            ('est_pl_params',         'fit_static_point_lens'),
            ('fit_pspl',              'fit_static_point_lens'),
            ('fit_parallax_u0_plus',  'fit_point_lens_parallax'),
            ('fit_parallax_u0_minus', 'fit_point_lens_parallax'),
            ('renormalize_datasets',  'renormalize'),
            ('refit_all',             'renormalize'),
        ]
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_stop_after_fit_binary(self):
        """
        stop_after='fit_binary' halts after fit_binary_models.
        """
        fitter = self._make_fitter(stop_after='fit_binary')

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = [
            ('run_ef_grid',                  'event_search'),
            ('est_pl_params',                'fit_static_point_lens'),
            ('fit_pspl',                     'fit_static_point_lens'),
            ('fit_parallax_u0_plus',         'fit_point_lens_parallax'),
            ('fit_parallax_u0_minus',        'fit_point_lens_parallax'),
            ('renormalize_datasets',         'renormalize'),
            ('refit_all',                    'renormalize'),
            ('select_best_point_lens_model', 'search_for_anomaly'),
            ('compute_residuals',            'search_for_anomaly'),
            ('run_af_grid',                  'search_for_anomaly'),
            ('est_binary_params',            'search_for_anomaly'),
            ('fit_binary_models',            'fit_binary'),
        ]
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_resume_after_stop_before_fit_binary(self):
        """
        When completed_steps ends at search_for_anomaly,
        planned_steps starts at fit_binary_models.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(EXPECTED_STEPS_BINARY[:11])
        fitter.fit()

        expected = [
            ('fit_binary_models',          'fit_binary'),
            ('check_needs_renorm',         'check_binary_renorm'),
            ('run_parallax_grid_u0_plus',  'run_parallax_grids'),
            ('run_parallax_grid_u0_minus', 'run_parallax_grids'),
        ]
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_resume_after_stop_after_check_binary_renorm(self):
        """
        When completed_steps ends at check_binary_renorm,
        planned_steps contains only the parallax grid steps.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(EXPECTED_STEPS_BINARY[:13])
        fitter.fit()

        expected = [
            ('run_parallax_grid_u0_plus',  'run_parallax_grids'),
            ('run_parallax_grid_u0_minus', 'run_parallax_grids'),
        ]
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_check_needs_renorm_inserts_steps_when_true(self):
        """
        When check_needs_renorm returns dynamic steps, renormalize_datasets
        and refit_all are inserted and executed before run_parallax_grids.
        """
        fitter = self._make_fitter()
        fitter.completed_steps = _make_noop_steps(EXPECTED_STEPS_BINARY[:12])

        dynamic_steps = [
            WorkflowStep(
                name='renormalize_datasets',
                stage='check_binary_renorm',
                action=MagicMock(return_value=None),
                description='No-op renormalize_datasets',
            ),
            WorkflowStep(
                name='refit_all',
                stage='check_binary_renorm',
                action=MagicMock(return_value=None),
                description='No-op refit_all',
            ),
        ]

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(fitter, 'check_needs_renorm',
                             return_value=dynamic_steps))
            stack.enter_context(
                patch.object(fitter, 'run_parallax_grid', return_value=None))
            fitter.fit()

        expected = [
            ('check_needs_renorm',        'check_binary_renorm'),
            ('renormalize_datasets',      'check_binary_renorm'),
            ('refit_all',                 'check_binary_renorm'),
            ('run_parallax_grid_u0_plus', 'run_parallax_grids'),
            ('run_parallax_grid_u0_minus','run_parallax_grids'),
        ]
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)


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
            fit_type='point lens',
            renormalize_errors=False,
            initial_results=INITIAL_RESULTS)
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        stack = ExitStack()
        mocks = {}
        mocks['run_ef_grid'] = stack.enter_context(
            patch.object(fitter, 'run_ef_grid', return_value={}))
        mocks['est_pl_params'] = stack.enter_context(
            patch.object(fitter, 'est_pl_params', return_value={}))
        mocks['fit_pspl'] = stack.enter_context(
            patch.object(fitter, 'fit_pspl', return_value=None))
        mocks['fit_parallax'] = stack.enter_context(
            patch.object(fitter, 'fit_parallax', return_value=None))
        stack.mocks = mocks
        return stack

    def test_dry_run_skips_est_pl_params(self):
        """
        When a static PSPL is supplied via initial_results with
        fit_type='point lens', planned_steps starts at fit_pspl,
        skipping est_pl_params.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.fit()

        expected = [
            ('fit_pspl',              'fit_static_point_lens'),
            ('fit_parallax_u0_plus',  'fit_point_lens_parallax'),
            ('fit_parallax_u0_minus', 'fit_point_lens_parallax'),
        ]
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_fit_pspl_uses_supplied_params_as_seed(self):
        """
        fit_pspl is called with the user-supplied PSPL params as seed.
        """
        fitter = self._make_fitter()

        stack = self._patch_fit_methods(fitter)
        with stack:
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
            fit_type='binary lens',
            renormalize_errors=False,
            initial_results=INITIAL_RESULTS)
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        stack = ExitStack()
        mocks = {}
        mocks['select_best_point_lens_model'] = stack.enter_context(
            patch.object(fitter, 'select_best_point_lens_model',
                         return_value=None))
        mocks['compute_residuals'] = stack.enter_context(
            patch.object(fitter, 'compute_residuals', return_value=None))
        mocks['run_af_grid'] = stack.enter_context(
            patch.object(fitter, 'run_af_grid', return_value=None))
        mocks['est_binary_params'] = stack.enter_context(
            patch.object(fitter, 'est_binary_params', return_value=None))
        mocks['fit_binary_models'] = stack.enter_context(
            patch.object(fitter, 'fit_binary_models', return_value=None))
        mocks['check_needs_renorm'] = stack.enter_context(
            patch.object(fitter, 'check_needs_renorm', return_value=None))
        stack.mocks = mocks
        return stack

    def test_dry_run_starts_at_search_for_anomaly(self):
        """
        When a static PSPL is supplied via initial_results with
        fit_type='binary lens', planned_steps starts at search_for_anomaly,
        skipping all point-lens stages.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.fit()

        expected = [
            ('select_best_point_lens_model', 'search_for_anomaly'),
            ('compute_residuals',            'search_for_anomaly'),
            ('run_af_grid',                  'search_for_anomaly'),
            ('est_binary_params',            'search_for_anomaly'),
            ('fit_binary_models',            'fit_binary'),
            ('check_needs_renorm',           'check_binary_renorm'),
        ]
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
            fit_type='binary lens',
            renormalize_errors=True)
        defaults.update(kwargs)
        return MMEXOFASTFitter(restart_file=restart_file, **defaults)

    def test_binary_steps_added_after_point_lens_restart(self):
        """
        Restarting from a completed point-lens run (renormalize_errors=True,
        parallax_grid=False) with fit_type='binary lens' produces a step
        queue that starts at search_for_anomaly.
        """
        # Simulate completed point-lens run through refit_all
        pl_completed = _make_noop_steps([
            ('run_ef_grid',           'event_search'),
            ('est_pl_params',         'fit_static_point_lens'),
            ('fit_pspl',              'fit_static_point_lens'),
            ('fit_parallax_u0_plus',  'fit_point_lens_parallax'),
            ('fit_parallax_u0_minus', 'fit_point_lens_parallax'),
            ('renormalize_datasets',  'renormalize'),
            ('refit_all',             'renormalize'),
        ])

        pkl_path = os.path.join(self.tmp_path, 'fake.pkl')
        _make_fake_pickle(pkl_path, pl_completed)

        fitter = self._make_fitter(
            restart_file=pkl_path,
            dry_run=True)
        fitter.fit()

        expected = [
            ('select_best_point_lens_model', 'search_for_anomaly'),
            ('compute_residuals',            'search_for_anomaly'),
            ('run_af_grid',                  'search_for_anomaly'),
            ('est_binary_params',            'search_for_anomaly'),
            ('fit_binary_models',            'fit_binary'),
            ('check_needs_renorm',           'check_binary_renorm'),
        ]
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)