# test_workflow.py

import os.path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import exozippy
from exozippy.mmexofast import MMEXOFASTFitter, OutputConfig, WorkflowStep

GROUND_DATA_FILES = [os.path.join(
    exozippy.MULENS_DATA_PATH, 'OB140939',
    'n20100310.I.OGLE.OB140939.txt')]

COORDS = '17:47:12.25 -21:22:58.7'

# (name, stage, completes_stage)
EXPECTED_STEPS = [
    ('run_ef_grid',           'event_search',            True),
    ('est_pl_params',         'fit_static_point_lens',   False),
    ('fit_pspl',              'fit_static_point_lens',   True),
    ('fit_parallax_u0_plus',  'fit_point_lens_parallax', False),
    ('fit_parallax_u0_minus', 'fit_point_lens_parallax', True),
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
            renormalize_errors=False,
            verbose=True,
            output_config=OutputConfig(
                base_dir=self.tmp_path,
                file_head='ob0939_test',
                save_log=True,
                save_plots=False,
                save_latex_tables=False,
                save_restart_files=True))
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _make_noop_steps(self):
        """
        Build a WorkflowStep list matching the expected step definitions
        but with no-op actions. Used to test stop logic without running
        actual fits.
        """
        return [
            WorkflowStep(
                name=name,
                stage=stage,
                completes_stage=completes_stage,
                action=MagicMock(return_value=None),
                description=f'No-op for {name}',
            )
            for name, stage, completes_stage in EXPECTED_STEPS
        ]

    # --- dry run ---

    def test_dry_run_planned_steps(self):
        """
        Dry run for ground-only point-lens fit with renormalize_errors=False
        produces the expected step queue with correct names and stages.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.fit()

        expected = [(name, stage) for name, stage, _ in EXPECTED_STEPS]
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    # --- stop_before stage:step ---

    def test_stop_before_mid_stage_step_completed_steps(self):
        """
        stop_before='fit_static_point_lens:est_pl_params' executes only
        steps before est_pl_params (i.e. the event_search stage only).
        """
        fitter = self._make_fitter(
            stop_before='fit_static_point_lens:est_pl_params')

        with patch.object(fitter, '_build_remaining_steps',
                          return_value=self._make_noop_steps()):
            fitter.fit()

        expected = [
            ('run_ef_grid', 'event_search'),
        ]
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_stop_before_later_mid_stage_step_completed_steps(self):
        """
        stop_before='fit_static_point_lens:fit_pspl' executes steps through
        est_pl_params but not fit_pspl.
        """
        fitter = self._make_fitter(
            stop_before='fit_static_point_lens:fit_pspl')

        with patch.object(fitter, '_build_remaining_steps',
                          return_value=self._make_noop_steps()):
            fitter.fit()

        expected = [
            ('run_ef_grid',   'event_search'),
            ('est_pl_params', 'fit_static_point_lens'),
        ]
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    # --- stop_after stage:step ---

    def test_stop_after_mid_stage_step_completed_steps(self):
        """
        stop_after='fit_static_point_lens:est_pl_params' executes steps
        through and including est_pl_params.
        """
        fitter = self._make_fitter(
            stop_after='fit_static_point_lens:est_pl_params')

        with patch.object(fitter, '_build_remaining_steps',
                          return_value=self._make_noop_steps()):
            fitter.fit()

        expected = [
            ('run_ef_grid',   'event_search'),
            ('est_pl_params', 'fit_static_point_lens'),
        ]
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_stop_after_later_mid_stage_step_completed_steps(self):
        """
        stop_after='fit_static_point_lens:fit_pspl' executes steps through
        and including fit_pspl.
        """
        fitter = self._make_fitter(
            stop_after='fit_static_point_lens:fit_pspl')

        with patch.object(fitter, '_build_remaining_steps',
                          return_value=self._make_noop_steps()):
            fitter.fit()

        expected = [
            ('run_ef_grid',   'event_search'),
            ('est_pl_params', 'fit_static_point_lens'),
            ('fit_pspl',      'fit_static_point_lens'),
        ]
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    # --- stop_before stage-only ---

    def test_stop_before_stage_completed_steps(self):
        """
        stop_before='fit_static_point_lens' halts before the first step
        of that stage.
        """
        fitter = self._make_fitter(
            stop_before='fit_static_point_lens')

        with patch.object(fitter, '_build_remaining_steps',
                          return_value=self._make_noop_steps()):
            fitter.fit()

        expected = [
            ('run_ef_grid', 'event_search'),
        ]
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    # --- stop_after stage-only ---

    def test_stop_after_stage_completed_steps(self):
        """
        stop_after='fit_static_point_lens' halts after the last step
        of that stage (i.e. after fit_pspl).
        """
        fitter = self._make_fitter(
            stop_after='fit_static_point_lens')

        with patch.object(fitter, '_build_remaining_steps',
                          return_value=self._make_noop_steps()):
            fitter.fit()

        expected = [
            ('run_ef_grid',   'event_search'),
            ('est_pl_params', 'fit_static_point_lens'),
            ('fit_pspl',      'fit_static_point_lens'),
        ]
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)
