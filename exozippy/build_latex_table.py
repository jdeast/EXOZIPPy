# recursively traverse the event dictionary, find Parameters, and execute their corresponding latex commands
from parameter import Parameter
import pathlib

def build_latex_table(event, var_filename="variables.tex", table_filename="table.tex", caption=None):
    # delete previous files if they exist
    pathlib.Path(var_filename).unlink(missing_ok=True)
    pathlib.Path(table_filename).unlink(missing_ok=True)
    
    # write a header to the table
    with open(table_filename, 'a') as f: 
        f.write(r'\documentclass{aastex62}' + '\n')
        f.write(r'\usepackage{apjfonts}' + '\n')
        f.write(r'\begin{document}' + '\n')
        f.write(r'\startlongtable' + '\n')
        f.write(r'\begin{deluxetable*}{lcc}' + '\n')
        if caption != None: f.write(r'\tablecaption{' + caption + '}' + '\n')
        f.write(r'\tablehead{\colhead{~~~Parameter} & \colhead{Description} & \colhead{Values}}' + '\n')
        f.write(r'\startdata' + '\n')

    _build_latex_table(event, var_filename=var_filename, table_filename=table_filename)

    # write a footer to the table
    with open(table_filename, 'a') as f: 
        f.write(r'\enddata' + '\n')
        f.write(r'\end{deluxetable*}' + '\n')
        f.write(r'\bibliographystyle{apj}' + '\n')
        f.write(r'\bibliography{References}' + '\n')
        f.write(r'\end{document}' + '\n')

# recursively traverse the event dictionary, find Parameters, and execute their corresponding latex commands
def _build_latex_table(event,var_filename="variables.tex", table_filename="table.tex"):
    # if a dictionary, traverse through each key
    if isinstance(event, dict):
        for key, value in event.items():
            # if a nested dictionary, recurse
            if isinstance(value, dict):
                for d in _build_latex_table(value, var_filename=var_filename, table_filename=table_filename):
                    return d
            # if a list, loop through each element
            elif isinstance(value, list) or isinstance(value, tuple):
                try:
                    for v in value:
                        for d in _build_latex_table(v, var_filename=var_filename, table_filename=table_filename):
                            return d
                except:
                    pass
            # if a parameter, print its value (both to csv and tex files)
            elif isinstance(value, Parameter):
                with open(var_filename, 'a') as f: f.write(value.to_latex_var())
                with open(table_filename, 'a') as f: f.write(value.to_table_line())
    # if a parameter, print its value (both to csv and tex files)
    elif isinstance(event, Parameter):
        with open(var_filename, 'a') as f: f.write(event.to_latex_var())
        with open(table_filename, 'a') as f: f.write(event.to_table_line())
