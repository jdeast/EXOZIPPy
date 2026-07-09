import csv
import pathlib
import numpy as np
from ..components import Parameter


def _instance_count(p):
    return np.prod(p.shape).astype(int) if p.shape != () else 1


def _instance_name(params, index):
    """Return the named label for instance ``index``, or None if unavailable."""
    for p in params:
        if p.names and index < len(p.names):
            return str(p.names[index])
    return None


def _instance_subhead(name):
    """A secondary row that acts as an indented instance sub-header."""
    return (
        r"\multicolumn{4}{l}{~~~~~\textit{" + name + r":}} \\" + "\n"
    )


def build_csv_output(system, csv_filename):
    """Write a machine-readable CSV of posterior results.

    Comment header line lists columns: parname, value, up_err, low_err

    One row per parameter instance. Fixed parameters have empty error columns.
    """
    rows = []
    for comp in system.get_all_components():
        comp_params = [v for v in comp.__dict__.values() if isinstance(v, Parameter)]
        printable = [p for p in comp_params if p.print_to_table]
        for p in printable:
            n_instances = _instance_count(p)
            if p.posterior is not None and p.summary is None:
                p.compute_summary()

            if n_instances == 1:
                name = p.label
                if p.summary is not None:
                    med, ep, em = p.summary.format(sigfigs=2)
                    rows.append((name, med, ep, em))
                elif p.initval is not None:
                    val = float(np.atleast_1d(p.from_internal(p.initval))[0])
                    rows.append((name, val, "", ""))
            else:
                for i in range(n_instances):
                    name = p.get_display_label(i)
                    if p.summary is not None:
                        summ_list = p.summary if isinstance(p.summary, list) else [p.summary]
                        s = summ_list[i] if i < len(summ_list) else summ_list[-1]
                        med, ep, em = s.format(sigfigs=2)
                        rows.append((name, med, ep, em))
                    elif p.initval is not None:
                        inits = np.atleast_1d(p.from_internal(p.initval))
                        val = float(inits[i] if i < len(inits) else inits[-1])
                        rows.append((name, val, "", ""))

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        f.write("# parname, value, up_err, low_err\n")
        writer.writerows(rows)


def build_latex_output(system, var_filename="variables.tex", template_filename="table_template.tex",
                       caption=None, tablecomments=None):

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

        # All \newcommand defs span every index — emit them once per parameter
        for p in printable:
            all_defs.append(p.to_latex_def())
            all_defs.append(p.to_latex_prior_def())

        n_instances = max(_instance_count(p) for p in printable)

        if n_instances == 1:
            all_table_lines.append(rf"\sidehead{{{comp_label}:}}" + "\n")
            for p in printable:
                all_table_lines.append(p.to_table_line(note_mark=_mark_for(p)))
        else:
            # Component-level header (no instance name — sub-headers carry that)
            all_table_lines.append(rf"\sidehead{{{comp_label}:}}" + "\n")

            for i in range(n_instances):
                name = _instance_name(printable, i) or chr(ord("A") + i)
                all_table_lines.append(_instance_subhead(name))

                for p in printable:
                    p_n = _instance_count(p)
                    if p_n > 1:
                        all_table_lines.append(p.to_table_line_at(i, note_mark=_mark_for(p)))
                    elif i == 0:
                        # Scalar param shared across instances: show once
                        all_table_lines.append(p.to_table_line(note_mark=_mark_for(p)))

    with open(var_filename, 'w') as f:
        f.write(f"% ExoZIPPy Generated Variables - {system.name}\n")
        f.writelines(all_defs)

    with open(template_filename, 'w') as f:
        f.write(r"\documentclass{aastex701}" + "\n")
        f.write(r"\usepackage{apjfonts}" + "\n")
        f.write(rf"\input{{{pathlib.Path(var_filename).stem}}}" + "\n")
        f.write(r"\begin{document}" + "\n")
        f.write(r"\startlongtable" + "\n")
        f.write(r"\begin{deluxetable*}{lccc}" + "\n")
        if caption is not None:
            f.write(rf"\tablecaption{{{caption} \label{{tab:{system.name}}}}}" + "\n")
        f.write(r"\tablehead{\colhead{~~~Parameter} & \colhead{Description} & \colhead{Value} & \colhead{Prior}}" + "\n")
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
