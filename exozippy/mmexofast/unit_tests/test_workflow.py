# test_workflow.py

import os.path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import exozippy
from exozippy.mmexofast import MMEXOFASTFitter, WorkflowStep

GROUND_DATA_FILES = [os.path.join(
    exozippy.MULENS_DATA_PATH, 'OB140939',
    'n20100310.I.OGLE.OB140939.txt')]

COORDS = '17:47:12.25 -21:22:58.7'

# (name, stage)
EXPECTED_STEPS = [
    ('run_ef_grid',           'event_search'),
    ('est_pl_params',         'fit_static_point_lens'),
    ('fit_pspl',              'fit_static_point_lens'),
    ('fit_parallax_u0_plus',  'fit_point_lens_parallax'),
    ('fit_parallax_u0_minus', 'fit_point_lens_parallax'),
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

    def _make_noop_steps(self):
        """
        Build a WorkflowStep list matching the expected step definitions
        but with no-op actions. Used to populate completed_steps in
        resume tests without running real fits.
        """
        return [
            WorkflowStep(
                name=name,
                stage=stage,
                action=MagicMock(return_value=None),
                description=f'No-op for {name}',
            )
            for name, stage in EXPECTED_STEPS
        ]

    def _patch_fit_methods(self, fitter):
        """
        Context manager that patches all fit action methods with no-ops
        so the execution loop runs without performing real fits.

        Assumes:
          - completed_steps stores WorkflowStep objects
          - fit() does NOT reset completed_steps at the start of each call
          - _build_remaining_steps() uses completed_steps to trim the queue
        """
        stack = ExitStack()
        stack.enter_context(
            patch.object(fitter, 'run_ef_grid', return_value={}))
        stack.enter_context(
            patch.object(fitter, 'est_pl_params', return_value={}))
        stack.enter_context(
            patch.object(fitter, 'fit_pspl', return_value=None))
        stack.enter_context(
            patch.object(fitter, 'fit_parallax', return_value=None))
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
        noop_steps = self._make_noop_steps()
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = noop_steps[:1]  # run_ef_grid only
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
        noop_steps = self._make_noop_steps()
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = noop_steps[:2]  # run_ef_grid + est_pl_params
        fitter.fit()

        expected = [
            ('fit_pspl',              'fit_static_point_lens'),
            ('fit_parallax_u0_plus',  'fit_point_lens_parallax'),
            ('fit_parallax_u0_minus', 'fit_point_lens_parallax'),
        ]
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)
