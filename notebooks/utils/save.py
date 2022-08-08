import datetime, json, glob, re
import pandas as pd
from json import JSONDecodeError


def get_timestamp(strftime="%y%m%dT%H%M%S"):
    x = datetime.datetime.now()
    return x.strftime(strftime)


def save_as_json(d:dict, path):
    path = path + (".json" if path[-5:] != ".json" else "")
    with open(path, 'w') as file:
        json.dump(d, file, indent=4, sort_keys=True)
        print(f'saving json at: {path}')


def append_json_array(d:dict, path, array_key="results"):

    # create file if it does not exist
    try:
        with open(path, "r+") as file:
            pass
    except FileNotFoundError:
        with open(path, "w") as file:
            pass
    
    # now that it exists append a new object on array 
    with open(path, "r+") as file:
        # if file is empty create a new dictionary
        try:
            json_dict = json.load(file)
        except JSONDecodeError:
            json_dict = {array_key: []}
        
        json_dict[array_key].append(d)
        
        # set file writing pointer at the beginning (offset 0)
        file.seek(0)
        json.dump(json_dict, file, indent=4, sort_keys=True)
    
    # except FileNotFoundError:


def load_json_array(path, array_key="results"):

    with open(path, "r") as file:
        json_dict = json.load(file)
    return pd.json_normalize(json_dict[array_key])


def get_path_from_re(re):
    return sorted(glob.glob(re))[-1]


def get_version(path, logdir="lightning_logs", version_pattern=r"version_(\d)+"):
    dirs = sorted(glob.glob(f"{path}/{logdir}/*"))
    if len(dirs) > 0:
        version_dirs = [re.search(version_pattern, d).group(0) for d in dirs]
        latest_version = int([re.search(r"\d+", d).group(0) for d in version_dirs][-1])
        return f'version_{latest_version + 1:02}'
    else:
        return f'version_00'
