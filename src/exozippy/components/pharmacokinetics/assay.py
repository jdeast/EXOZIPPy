"""Measured concentrations, the residual error model, and the likelihood.

READ README.md IN THIS DIRECTORY FIRST.
"""

import logging
import os

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from astropy import units as u

from ..component import Component
from . import physics

logger = logging.getLogger(__name__)

# Internal units.  The Parameter layer converts every PARAMETER for us; these
# are for the DATA columns, which no Parameter owns.
_CONC = u.mg / u.L
_TIME = u.hr


class Assay(Component):
    """One file of measured concentrations, and the likelihood over it.

    **This component was written by an astrophysicist and an LLM. No biologist,
    pharmacologist, clinician, or pharmacometrician has reviewed it.** It
    exists to demonstrate and enforce the component-agnostic architecture and
    to be a starting point for non-astronomy development. See ``README.md`` in
    this directory before relying on any of it.

    One instance per data file::

        assay:
          - name: theoph
            datafile: theoph.csv
            columns: {subject: 0, time: 3, obs: 4}
            conc_unit: "mg/L"
            time_unit: "hr"

    THE FILE HOLDS EVERY SUBJECT, which is how the field stores these data and
    is why ``build_maps`` reads a subject COLUMN rather than the component
    taking one file per individual. That column is resolved against
    ``subject``'s instance names, so the map is built from names the user
    already wrote rather than from row order.

    ``subject_prefix`` exists because of a real collision between this field's
    conventions and the core's. Clinical data label subjects ``1, 2, 3...``,
    but ``config.validate_instance_names`` rejects a purely numeric instance
    name -- it would be ambiguous with the internal ``subject.0`` index
    notation. So the subjects are declared ``S1, S2, ...`` and the prefix is
    applied to the data column before matching. An explicit key rather than a
    fallback that tries the bare name and then a prefixed one: a silent second
    attempt would pair the wrong rows whenever both spellings exist.

    WHY THIS DOES NOT INHERIT ``Instrument``. ``Instrument`` is the shared
    scaffolding for a data component and most of it -- columns, masks,
    detrending, GP kernels, robust likelihoods, jitter -- is field-neutral and
    would fit here. Its VOCABULARY is not: ``time_frame`` defaults to ``bjd``,
    ``time_offset`` is in days, times are converted through ``_to_bjd_tdb``,
    and ``time_location`` is an observatory. Those defaults pass a blood draw
    through untouched, which is precisely the trap -- it would work while
    documenting a barycentric correction on a plasma sample. Inheriting
    ``Component`` directly is also the experiment: ``components.md`` declares
    the extension API to be ``Component`` + ``Parameter`` + the manifest
    vocabulary + the four-file layout, and this component is a test of that
    sentence as written. What it re-implements as a result is the finding, and
    is recorded in ``pharmacokinetics.md``.
    """

    label = "Assay"

    @property
    def prefix(self):
        return "assay"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "datafile",
                "kind": "datafile",
                "accepts": "*.csv",
                "required": True,
                "doc": (
                    "CSV of measured concentrations, one row per sample, "
                    "with a subject identifier, a time and a concentration "
                    "column."
                ),
            },
            {
                "key": "columns",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Column indices or header names: "
                    "{subject: <col>, time: <col>, obs: <col>}. "
                    "Defaults to {subject: 0, time: 1, obs: 2}."
                ),
            },
            {
                "key": "subject_prefix",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "String prepended to each value of the subject column "
                    "before matching it against the declared 'subject:' "
                    "names. Clinical data label subjects 1, 2, 3..., but a "
                    "purely numeric instance name is rejected by the core "
                    "(it would collide with the internal index notation), so "
                    "subjects declared as 'S1', 'S2' are matched to data "
                    "rows '1', '2' with subject_prefix: 'S'."
                ),
            },
            {
                "key": "conc_unit",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Unit of the concentration column, e.g. 'mg/L' "
                    "(default), 'ug/mL' (identical), or 'ng/mL' (1000x)."
                ),
            },
            {
                "key": "time_unit",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": "Unit of the time column. Default 'hr'.",
            },
        ]

    @classmethod
    def get_utilities(cls):
        """The Theophylline fetcher, declared by the component that needs it.

        The data are NOT redistributed with EXOZIPPy -- see fetch_theoph.py
        for the licence reasoning -- so the example fetches them on request.
        Declared here rather than wired into the CLI so the GUI and any other
        consumer discover it generically, with no component name hardcoded
        outside component-owned code.
        """
        from ...utilities.registry import (
            UtilitySpec,
            argparse_subprocess_runner,
        )
        from . import fetch_theoph

        return [
            UtilitySpec(
                name="fetch_theoph",
                label="Download Theophylline example data",
                description=(
                    "Fetch the Theophylline pharmacokinetic dataset "
                    "(Boeckmann, Sheiner & Beal 1994) used by "
                    "examples/theophylline. Not redistributed with EXOZIPPy."
                ),
                component_keys=["assay"],
                available=True,
                build_parser=fetch_theoph.build_parser,
                run=argparse_subprocess_runner(
                    "exozippy.components.pharmacokinetics.fetch_theoph"
                ),
            ),
        ]

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def load_data(self, system):
        """Read each file, convert units, and record which subject each row is.

        Concentrations are converted to mg/L and times to hours HERE, because
        no ``Parameter`` owns a data column and so nothing else would. A
        ``conc_unit: ng/mL`` is a factor of 1000 -- the size of error this
        codebase has hidden before -- so the conversion is explicit and its
        factor is logged.
        """
        self.time = []
        self.obs = []
        self.subject_names = []

        for cfg, name in zip(self.config, self.names):
            where = f"{self.prefix} '{name}'"
            frame = self._read_file(cfg, where)
            cols = self._resolve_columns(cfg, frame, where)

            time = self._to_internal(
                frame.iloc[:, cols["time"]].to_numpy(dtype=float),
                cfg.get("time_unit", "hr"),
                _TIME,
                f"{where} time",
            )
            obs = self._to_internal(
                frame.iloc[:, cols["obs"]].to_numpy(dtype=float),
                cfg.get("conc_unit", "mg/L"),
                _CONC,
                f"{where} concentration",
            )
            prefix = str(cfg.get("subject_prefix", ""))
            subjects = [
                prefix + str(s).strip()
                for s in frame.iloc[:, cols["subject"]].to_numpy()
            ]

            good = np.isfinite(time) & np.isfinite(obs)
            if not good.any():
                raise ValueError(
                    f"[{where}] no finite (time, concentration) rows were "
                    f"read from {cfg.get('datafile')!r}. Check the 'columns:' "
                    f"mapping."
                )
            if not good.all():
                logger.warning(
                    "[%s] dropped %d of %d rows with a non-finite time or "
                    "concentration.",
                    where,
                    int((~good).sum()),
                    good.size,
                )

            self.time.append(time[good])
            self.obs.append(obs[good])
            self.subject_names.append(
                [s for s, keep in zip(subjects, good) if keep]
            )

        self.all_time = np.concatenate(self.time)
        self.all_obs = np.concatenate(self.obs)

        # Captured at stage 1 because build_maps (stage 2) needs it and takes
        # no `system`.  The names come from the parsed config, not from
        # subject's own load_data, so there is no ordering dependency between
        # the two components' stage-1 passes.
        self._subject_names_declared = list(system.subject.names)

    def _read_file(self, cfg, where):
        """The CSV as a DataFrame, with a readable error when it is not there."""
        path = cfg.get("datafile")
        if not path:
            raise ValueError(f"[{where}] a 'datafile:' is required.")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{where}] datafile {path!r} does not exist. The "
                f"Theophylline example fetches its data on first run -- see "
                f"'exozippy-fetch-theoph' and this component's README."
            )
        return pd.read_csv(path)

    @staticmethod
    def _resolve_columns(cfg, frame, where):
        """Map each role to a positional column index.

        Accepts an index or a header NAME per role, because these files
        routinely ship with headers and an index is the thing that silently
        reads the wrong column when a file gains one.
        """
        spec = dict(cfg.get("columns") or {})
        defaults = {"subject": 0, "time": 1, "obs": 2}
        resolved = {}

        for role, fallback in defaults.items():
            raw = spec.pop(role, fallback)
            if isinstance(raw, str):
                if raw not in frame.columns:
                    raise ValueError(
                        f"[{where}] columns.{role}: {raw!r} is not a column "
                        f"of the file. Available: {list(frame.columns)}."
                    )
                resolved[role] = frame.columns.get_loc(raw)
            else:
                idx = int(raw)
                if not 0 <= idx < frame.shape[1]:
                    raise ValueError(
                        f"[{where}] columns.{role}: {idx} is out of range for "
                        f"a file with {frame.shape[1]} columns."
                    )
                resolved[role] = idx

        if spec:
            raise ValueError(
                f"[{where}] unknown entries in 'columns:': {sorted(spec)}. "
                f"Valid roles are {sorted(defaults)}."
            )
        if len(set(resolved.values())) != len(resolved):
            raise ValueError(
                f"[{where}] 'columns:' maps two roles to one column: "
                f"{resolved}."
            )
        return resolved

    @staticmethod
    def _to_internal(values, unit_str, target, where):
        """Convert a data column into its internal unit, loudly."""
        try:
            unit = u.Unit(str(unit_str))
        except Exception:
            raise ValueError(
                f"[{where}] {unit_str!r} is not a unit astropy understands."
            ) from None
        if not unit.is_equivalent(target):
            raise ValueError(
                f"[{where}] unit {unit_str!r} is not convertible to "
                f"{target}. Check the column mapping and the unit."
            )
        factor = float((1.0 * unit).to(target).value)
        if factor != 1.0:
            logger.info(
                "[%s] converting %s -> %s (x%g).", where, unit, target, factor
            )
        return values * factor

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def build_maps(self):
        """``subject_map``: which subject each observation belongs to.

        Resolved against ``subject``'s declared instance NAMES, so the pairing
        comes from what the user wrote rather than from the order rows happen
        to appear in. A name in the data with no matching ``subject:`` block
        raises: silently dropping those rows would fit a subset of the data
        and report it as the whole.
        """
        names = list(self._subject_names_declared)
        index = {name: i for i, name in enumerate(names)}

        rows = []
        unknown = set()
        for per_file in self.subject_names:
            for name in per_file:
                if name not in index:
                    unknown.add(name)
                else:
                    rows.append(index[name])

        if unknown:
            raise ValueError(
                f"[{self.prefix}] the data name subjects with no 'subject:' "
                f"block: {sorted(unknown)}. Declared subjects are {names}. "
                f"Add a block for each, or correct the subject column. If the "
                f"data label subjects numerically (1, 2, 3...), note that a "
                f"purely numeric instance name is rejected by the core -- "
                f"declare them as 'S1', 'S2', ... and set "
                f"'subject_prefix: \"S\"' on this assay."
            )

        self.subject_map = np.asarray(rows, dtype=int)

        missing = set(range(len(names))) - set(rows)
        if missing:
            logger.warning(
                "[%s] subjects %s have no observations in any assay file; "
                "their parameters are constrained only by their priors.",
                self.prefix,
                [names[i] for i in sorted(missing)],
            )

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------

    def register_parameters(self, system):
        """Declare the residual error model, and seed it from the data.

        ``sigma_add`` is seeded through ``add_hint`` rather than a manifest
        ``initval`` option: a hint is a RANKED start value that the params
        file still beats, while a plain option is merged after ``resolve()``
        and would silently discard a value the user typed (config.md).
        """
        self.manifest = {"sigma_add": None, "sigma_prop": None}

        # PER ELEMENT, and per FILE.  The hint channel refuses the 2-part
        # broadcast form outright: a component knows which element it means,
        # and since elements may carry different `unit:` overrides there is no
        # single unit one scalar could be read in (config.py's
        # _reject_broadcast_hint_path).  Seeding from each file's own spread
        # is also the better answer -- two assays need not share a scale.
        #
        # A tenth of the observed spread: small enough that the proportional
        # term is not pre-empted, large enough not to start on the bound.
        for i, obs in enumerate(self.obs):
            if not (obs.size and np.isfinite(obs).any()):
                continue
            seed = 0.1 * float(np.nanstd(obs))
            if np.isfinite(seed) and seed > 0:
                self.config_manager.add_hint(
                    f"{self.prefix}.{i}.sigma_add", seed, rank=60
                )

    # ------------------------------------------------------------------
    # Stage 7
    # ------------------------------------------------------------------

    def predicted_concentration(self, system):
        """Model concentration at every observation, as one flat vector.

        The subject parameters are gathered through ``subject_map``, which is
        the same shape as ``mann`` reading ``star.mass.value[star_map]``: a
        component reads another component's built Parameters at stage 7 and
        needs no manifest dependency to do it.
        """
        smap = self.subject_map_tensor
        subject = system.subject
        return physics.calc_pk_concentration(
            pt.as_tensor_variable(self.all_time),
            subject.dose.value[smap],
            subject.ka.value[smap],
            subject.ke.value[smap],
            subject.v.value[smap],
        )

    def build_likelihood(self, model, system):
        predicted = self.predicted_concentration(system)

        # NOT a pm.Deterministic.  `predicted` has one entry PER OBSERVATION,
        # so storing it would put n_obs x n_draws x n_chains floats in the
        # trace -- and, worse, every one of those entries reaches the corner
        # plot and the trace pages as if it were a parameter.  On the 132-row
        # Theophylline example that was enough to get the wrap-up OOM-killed
        # after the fit itself had finished, with a screenful of "omitting
        # assay.predicted[k] (constant at 0)" for the t = 0 rows on the way
        # down.  No shipped data component stores one (transit's single
        # per-band dilution node is the only Deterministic among them), and
        # nothing needs it: the curve is a deterministic function of the
        # parameters, so a plotter recomputes it from a point.
        sigma = physics.combined_sigma(
            predicted, self.sigma_add.value, self.sigma_prop.value
        )
        pm.Normal(
            f"{self.prefix}.obs",
            mu=predicted,
            sigma=sigma,
            observed=self.all_obs,
        )

        self._add_prose(system)

    def _add_prose(self, system):
        from ...outputs.prose import get_collector

        prose = get_collector(system)
        n_obs = int(self.all_obs.size)
        n_files = self.n_elements
        prose.add(
            f"We fitted {n_obs} concentration measurements from {n_files} "
            + ("assay" if n_files == 1 else "assays")
            + ", with a combined residual error model whose standard "
            "deviation is $\\sqrt{\\sigma_{add}^2 + (\\sigma_{prop} C)^2}$ "
            "for a predicted concentration $C$.",
            section="data",
            key=f"{self.prefix}.inventory",
        )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
