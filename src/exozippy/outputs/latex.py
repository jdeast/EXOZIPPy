# recursively traverse the event dictionary, find Parameters, and execute their corresponding latex commands
#from exozippy.parameter import Parameter
import pathlib
from ..components import Parameter

def build_latex_output(system, var_filename="variables.tex", template_filename="table_template.tex",
                       caption=None):

    # 1. Collect all data from the OO hierarchy
    all_defs = []
    all_table_lines = []

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

    # 2. Iterate through components to build the table
    for comp in system.get_all_components():
        # Look inside the component's __dict__ to preserve Parameter order
        comp_params = [v for v in comp.__dict__.values() if isinstance(v, Parameter)]

        if comp_params:
            header_text = getattr(comp, "label", comp.__class__.__name__)
            all_table_lines.append(rf"\sidehead{{{header_text}}}" + "\n")

            for p in comp_params:
                if p.print_to_table:
                    all_defs.append(p.to_latex_def())
                    all_table_lines.append(p.to_table_line())

    # 2. Write the "Database" file (Just the \newcommand definitions)
    with open(var_filename, 'w') as f:
        f.write(f"% ExoZIPPy Generated Variables - {system.name}\n")
        f.writelines(all_defs)

    # 3. Write the "Template" file (The structural table)
    with open(template_filename, 'w') as f:
        f.write(r"\documentclass{aastex701}"+"\n")
        f.write(r"\usepackage{apjfonts}" + "\n")
        f.write(rf"\input{{{pathlib.Path(var_filename).stem}}}" + "\n")
        f.write(r"\begin{document}"+"\n")
        f.write(r"\startlongtable"+"\n")
        f.write(r"\begin{deluxetable*}{lccc}"+"\n")
        if caption != None: f.write(rf"\tablecaption{{{caption} \label{{tab:{system.name}}}}}" + "\n")
        f.write(r"\tablehead{\colhead{~~~Parameter} & \colhead{Description} & \colhead{Value} & \colhead{Prior}}" + "\n")
        f.write(r"\startdata"+"\n")

        # Insert the actual data lines
        f.writelines(all_table_lines)

        # footer
        f.write(r"\enddata" + "\n")
        f.write(r"\end{deluxetable*}" + "\n")
        f.write(r"\bibliographystyle{aasjournalv7}" + "\n")
        f.write(r"\bibliography{References}" + "\n")
        f.write(r"\end{document}" + "\n")
