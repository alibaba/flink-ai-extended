import os
from ai_flow.api.configuration import set_project_config_file
import types


def get_root_path():
    return os.path.dirname(os.path.abspath(__file__))


def get_project_config_file(path):
    return get_project_path(path) + "/project.yaml"


def get_master_config_file():
    return get_root_path() + "/master.yaml"


def set_project_config(path):
    print("Project config path: " + get_project_config_file(path))
    set_project_config_file(get_project_config_file(path))


def get_project_path(path):
    return get_root_path() + path


def list_all_fields(obj):
    for f in dir(obj):
        if str(f).startswith("_"):
            continue
        if isinstance(f, types.MethodType):
            continue
        print(str(f) + str(type(f)))
        if hasattr(obj, "__dict__") and f in obj.__dict__:
            list_all_fields(obj.__dict__[f])
    print("----------")
