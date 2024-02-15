import json
def read_parfile(filename):
    with open(filename) as f:
        contents = f.read()
    return json.loads(contents)
