import json
def read_parfile(filename):
    with open(filename) as f:
        contents = json.load(f)
    return contents
