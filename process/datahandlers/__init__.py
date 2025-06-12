from .darpa_handler import DARPAHandler
from .atlas_handler import ATLASHandler

__all__ = ["DARPAHandler", "ATLASHandler"]

handler_map = {
    "theia": DARPAHandler,
    "cadets": DARPAHandler,
    "clearscope": DARPAHandler,
    "trace": DARPAHandler,
    "atlas": ATLASHandler}

path_map = {
    "theia": "../data_files/theia",
    "cadets": "../data_files/cadets",
    "clearscope": "../data_files/clearscope",
    "trace": "../data_files/trace",
    "atlas": "../atlas_data",
}

def get_handler(name):
    cls = handler_map.get(name.lower())
    base_path = path_map.get(name)
    if base_path is None:
        raise ValueError(f"未配置数据路径: {name}")
    if cls is None:
        raise ValueError(f"未知数据集: {name}")
    return cls(base_path)
