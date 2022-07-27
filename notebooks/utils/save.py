import datetime, json, glob, re


def get_timestamp(strftime="%y%m%dT%H%M%S"):
    x = datetime.datetime.now()
    return x.strftime(strftime)


def save_as_json(d:dict, path):
    path = path + (".json" if path[-5:] != ".json" else "")
    with open(path, 'w') as file:
        json.dump(d, file, indent=4, sort_keys=True)
        print(f'saving json at: {path}')


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
