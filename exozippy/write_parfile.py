import json
def write_parfile(params,filename):
    with open(filename,'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
