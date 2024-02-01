# recursively traverse the event dictionary, find Parameters, and execute their corresponding latex commands
def build_latex_table(event, var_filename="variables.tex", table_filename="table.tex"):
    if isinstance(event, dict):
        for key, value in event.items():
            if isinstance(value, dict):
                for d in build_latex_table(value, var_filename=var_filename, table_filename=table_filename):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    for d in build_latex_table(v, var_filename=var_filename, table_filename=table_filename):
                        yield d
            else:
                if isinstance(value, Parameter):
                    with open(var_filename, 'a') as f: f.write(value.to_latex_var())
                    with open(table_filename, 'a') as f: f.write(value.to_table_line())
    else:
        if isinstance(value, Parameter):
            with open(var_filename, 'a') as f: f.write(value.to_latex_var())
            with open(table_filename, 'a') as f: f.write(value.to_table_line())
