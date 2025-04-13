# recursively traverse the event dictionary, find Parameters, and execute their corresponding latex commands
from .parameter import Parameter
import pathlib

def build_latex_table(event, var_filename="variables.tex", table_filename="table.tex", caption=None):
    # delete previous files if they exist
    pathlib.Path(var_filename).unlink(missing_ok=True)
    pathlib.Path(table_filename).unlink(missing_ok=True)
    
    # write a header to the table
    with open(table_filename, 'a') as f: 
        f.write("\documentclass{aastex62}\n")
        f.write("\\usepackage{apjfonts}\n")
        f.write("\\begin{document}\n")
        f.write("\startlongtable\n")
        f.write("\\begin{deluxetable*}{lcc}\n")
        if caption != None: f.write('\tablecaption{' + caption + '}')
        f.write("\\tablehead{\colhead{~~~Parameter} & \colhead{Description} & \colhead{Values}}\n")
        f.write("\startdata\n")

    _build_latex_table(event, var_filename=var_filename, table_filename=table_filename)

    # write a footer to the table
    with open(table_filename, 'a') as f: 
        f.write("\enddata\n")
        f.write("\end{deluxetable*}\n")
        f.write("\\bibliographystyle{apj}\n")
        f.write("\\bibliography{References}\n")
        f.write("\end{document}\n")

# recursively traverse the event dictionary, find Parameters, and execute their corresponding latex commands
def _build_latex_table(event,var_filename="variables.tex", table_filename="table.tex"):
    if isinstance(event, dict):
        for key, value in event.items():
            if isinstance(value, dict):
                for d in _build_latex_table(value, var_filename=var_filename, table_filename=table_filename):
                    return d
            elif isinstance(value, list) or isinstance(value, tuple):
                try:
                    for v in value:
                        for d in _build_latex_table(v, var_filename=var_filename, table_filename=table_filename):
                            return d
                except:
                    pass
            elif isinstance(value, Parameter):
                with open(var_filename, 'a') as f: f.write(value.to_latex_var())
                with open(table_filename, 'a') as f: f.write(value.to_table_line())
    elif isinstance(value, Parameter):
        with open(var_filename, 'a') as f: f.write(value.to_latex_var())
        with open(table_filename, 'a') as f: f.write(value.to_table_line())
