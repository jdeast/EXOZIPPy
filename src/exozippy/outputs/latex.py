import csv
import pathlib
import numpy as np
from ..components import Parameter
from ..components.parameter import _idx_to_words


def _instance_count(p):
    return np.prod(p.shape).astype(int) if p.shape != () else 1


def _instance_name(params, index):
    """Return the named label for instance ``index``, or None if unavailable."""
    for p in params:
        if p.names and index < len(p.names):
            return str(p.names[index])
    return None


def _instance_subhead(name, n_cols=4):
    """A secondary row that acts as an indented instance sub-header."""
    return (
        rf"\multicolumn{{{n_cols}}}{{l}}{{~~~~~\textit{{" + name + r":}} \\" + "\n"
    )


def _multimodal(mode_report):
    return mode_report is not None and mode_report.n_modes > 1


def _ensure_mode_summaries(system, p, mode_report):
    """Compute per-mode summaries for a sampled parameter if missing."""
    if p.posterior is None or p.mode_summaries is not None:
        return
    labels = getattr(system, "mode_labels", None)
    if labels is None:
        raise ValueError(
            "mode_report given but system has no mode labels; call "
            "identify_modes(idata) before system.distribute_posterior(idata)")
    p.compute_mode_summaries(labels, mode_report.n_modes)


def build_csv_output(system, csv_filename, mode_report=None):
    """Write a machine-readable CSV of posterior results.

    Comment header line lists columns: parname, value, up_err, low_err.
    With a multimodal ``mode_report``, two extra leading columns (mode,
    weight) are added; each parameter gets one row per mode plus a combined
    row (mode 'all', weight 1).  Fixed parameters have empty error columns.
    """
    multimodal = _multimodal(mode_report)

    rows = []
    for comp in system.get_all_components():
        comp_params = [v for v in comp.__dict__.values() if isinstance(v, Parameter)]
        printable = [p for p in comp_params if p.print_to_table]
        for p in printable:
            n_instances = _instance_count(p)
            if p.posterior is not None and p.summary is None:
                p.compute_summary()
            if multimodal:
                _ensure_mode_summaries(system, p, mode_report)

            def emit(name, index):
                """Rows for one parameter instance (one per mode if multimodal)."""
                def summ_at(summary):
                    s_list = summary if isinstance(summary, list) else [summary]
                    return s_list[index] if index < len(s_list) else s_list[-1]

                if p.summary is not None:
                    med, ep, em = summ_at(p.summary).format(sigfigs=2)
                    if multimodal:
                        rows.append((name, "all", 1.0, med, ep, em))
                        for k, m in enumerate(mode_report.modes):
                            med, ep, em = summ_at(p.mode_summaries[k]).format(sigfigs=2)
                            rows.append((name, k + 1, round(m.weight, 4), med, ep, em))
                    else:
                        rows.append((name, med, ep, em))
                elif p.initval is not None:
                    inits = np.atleast_1d(p.from_internal(p.initval))
                    val = float(inits[index] if index < len(inits) else inits[-1])
                    if multimodal:
                        rows.append((name, "all", 1.0, val, "", ""))
                    else:
                        rows.append((name, val, "", ""))

            if n_instances == 1:
                emit(p.label, 0)
            else:
                for i in range(n_instances):
                    emit(p.get_display_label(i), i)

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        if multimodal:
            f.write("# parname, mode, weight, value, up_err, low_err\n")
        else:
            f.write("# parname, value, up_err, low_err\n")
        writer.writerows(rows)


def build_latex_output(system, var_filename="variables.tex", template_filename="table_template.tex",
                       caption=None, tablecomments=None, mode_report=None):
    """Write the LaTeX variable definitions and deluxetable template.

    With a multimodal ``mode_report`` (from outputs.modes.identify_modes,
    after system.distribute_posterior picked up the mode labels), the table
    gets one Value column per mode, a leading mode-weight row, and the weight
    provenance appended to the table comments.  Per-mode macro names carry a
    ``modeone``/``modetwo``/... suffix so every mode can be cited in the same
    document; the unsuffixed macros keep their combined-posterior meaning.
    """
    multimodal = _multimodal(mode_report)
    mode_suffixes = ([f"mode{_idx_to_words(k + 1)}"
                      for k in range(mode_report.n_modes)]
                     if multimodal else None)

    all_defs = []
    all_table_lines = []

    # Distinct Parameter.table_note texts get sequential tablenotemark
    # letters; the matching \tablenotetext lines are emitted after \enddata.
    note_marks = {}

    def _mark_for(p):
        if not getattr(p, "table_note", None):
            return None
        if p.table_note not in note_marks:
            note_marks[p.table_note] = chr(ord("a") + len(note_marks))
        return note_marks[p.table_note]

    all_defs.append(r"\providecommand{\bjdtdb}{\ensuremath{\rm {BJD_{TDB}}}}" + "\n")
    all_defs.append(r"\providecommand{\feh}{\ensuremath{\left[{\rm Fe}/{\rm H}\right]}}" + "\n")
    all_defs.append(r"\providecommand{\teff}{\ensuremath{T_{\rm eff}}}" + "\n")
    all_defs.append(r"\providecommand{\teq}{\ensuremath{T_{\rm eq}}}" + "\n")
    all_defs.append(r"\providecommand{\ecosw}{\ensuremath{e\cos{\omega_*}}}" + "\n")
    all_defs.append(r"\providecommand{\esinw}{\ensuremath{e\sin{\omega_*}}}" + "\n")
    all_defs.append(r"\providecommand{\msun}{\ensuremath{\,M_\Sun}}" + "\n")
    all_defs.append(r"\providecommand{\rsun}{\ensuremath{\,R_\Sun}}" + "\n")
    all_defs.append(r"\providecommand{\lsun}{\ensuremath{\,L_\Sun}}" + "\n")
    all_defs.append(r"\providecommand{\mj}{\ensuremath{\,M_{\rm J}}}" + "\n")
    all_defs.append(r"\providecommand{\rj}{\ensuremath{\,R_{\rm J}}}" + "\n")
    all_defs.append(r"\providecommand{\me}{\ensuremath{\,M_{\rm E}}}" + "\n")
    all_defs.append(r"\providecommand{\re}{\ensuremath{\,R_{\rm E}}}" + "\n")
    all_defs.append(r"\providecommand{\fave}{\langle F \rangle}" + "\n")
    all_defs.append(r"\providecommand{\fluxcgs}{10$^9$ erg s$^{-1}$ cm$^{-2}$}" + "\n")

    for comp in system.get_all_components():
        comp_params = [v for v in comp.__dict__.values() if isinstance(v, Parameter)]
        if not comp_params:
            continue

        printable = [p for p in comp_params if p.print_to_table]
        if not printable:
            continue

        comp_label = getattr(comp, "label", comp.__class__.__name__)

        # All \newcommand defs span every index — emit them once per parameter.
        # When multimodal, the unsuffixed def is the pooled-across-modes
        # value; suppress it for sampled parameters (fixed parameters have no
        # per-mode variation, so their single def still applies to every mode).
        for p in printable:
            if multimodal:
                _ensure_mode_summaries(system, p, mode_report)
            if not multimodal or p.posterior is None:
                all_defs.append(p.to_latex_def())
            all_defs.append(p.to_latex_prior_def())
            if multimodal:
                all_defs.append(p.to_latex_mode_defs())

        n_instances = max(_instance_count(p) for p in printable)

        if n_instances == 1:
            all_table_lines.append(rf"\sidehead{{{comp_label}:}}" + "\n")
            for p in printable:
                all_table_lines.append(p.to_table_line(note_mark=_mark_for(p),
                                                       mode_suffixes=mode_suffixes))
        else:
            # Component-level header (no instance name — sub-headers carry that)
            all_table_lines.append(rf"\sidehead{{{comp_label}:}}" + "\n")

            for i in range(n_instances):
                name = _instance_name(printable, i) or chr(ord("A") + i)
                n_cols = 4 if not multimodal else 3 + mode_report.n_modes
                all_table_lines.append(_instance_subhead(name, n_cols=n_cols))

                for p in printable:
                    p_n = _instance_count(p)
                    if p_n > 1:
                        all_table_lines.append(p.to_table_line_at(i, note_mark=_mark_for(p),
                                                                  mode_suffixes=mode_suffixes))
                    elif i == 0:
                        # Scalar param shared across instances: show once
                        all_table_lines.append(p.to_table_line(note_mark=_mark_for(p),
                                                               mode_suffixes=mode_suffixes))

    if multimodal:
        # Mode weights are citable macros too, and lead the table as a row.
        for k, m in enumerate(mode_report.modes):
            all_defs.append(
                rf"\providecommand{{\ezmodeweight{_idx_to_words(k + 1)}}}"
                rf"{{\ensuremath{{{m.weight:.3f}}}}}" + "\n")
        weight_cells = " & ".join(
            rf"\ezmodeweight{_idx_to_words(k + 1)}\dotfill"
            for k in range(mode_report.n_modes))
        weight_row = (r"~~~~Mode weight\dotfill & "
                      r"Fraction of posterior mass\dotfill & "
                      + weight_cells + r" &  \\" + "\n")
        all_table_lines.insert(0, weight_row)

        provenance_note = (
            "Mode weights: " + mode_report.provenance + ". Combined "
            "(pooled-across-modes) parameter values are suppressed above "
            "because pooled values inherit the mode-weight provenance; see "
            "the per-mode columns.")
        tablecomments = (provenance_note if not tablecomments
                         else tablecomments + " " + provenance_note)

    if mode_report is not None and mode_report.n_invalid:
        invalid_note = (
            f"{mode_report.n_invalid} draws ({mode_report.invalid_frac:.2%}) "
            "rejected as numerically invalid -- this indicates a model or "
            "sampler bug; investigate before trusting this table.")
        tablecomments = (invalid_note if not tablecomments
                         else tablecomments + " " + invalid_note)

    with open(var_filename, 'w') as f:
        f.write(f"% ExoZIPPy Generated Variables - {system.name}\n")
        f.writelines(all_defs)

    n_value_cols = mode_report.n_modes if multimodal else 1
    colspec = "l" + "c" * (2 + n_value_cols)
    if multimodal:
        value_heads = " & ".join(rf"\colhead{{Value (mode {k + 1})}}"
                                 for k in range(mode_report.n_modes))
    else:
        value_heads = r"\colhead{Value}"

    with open(template_filename, 'w') as f:
        f.write(r"\documentclass{aastex701}" + "\n")
        f.write(r"\usepackage{apjfonts}" + "\n")
        f.write(rf"\input{{{pathlib.Path(var_filename).stem}}}" + "\n")
        f.write(r"\begin{document}" + "\n")
        f.write(r"\startlongtable" + "\n")
        f.write(rf"\begin{{deluxetable*}}{{{colspec}}}" + "\n")
        if caption is not None:
            f.write(rf"\tablecaption{{{caption} \label{{tab:{system.name}}}}}" + "\n")
        f.write(r"\tablehead{\colhead{~~~Parameter} & \colhead{Description} & "
                + value_heads + r" & \colhead{Prior}}" + "\n")
        f.write(r"\startdata" + "\n")
        f.writelines(all_table_lines)
        f.write(r"\enddata" + "\n")
        for text, mark in note_marks.items():
            f.write(rf"\tablenotetext{{{mark}}}{{{text}}}" + "\n")
        if tablecomments:
            f.write(rf"\tablecomments{{{tablecomments}}}" + "\n")
        f.write(r"\end{deluxetable*}" + "\n")
        f.write(r"\bibliographystyle{aasjournalv7}" + "\n")
        f.write(r"\bibliography{References}" + "\n")
        f.write(r"\end{document}" + "\n")
