from .darpa_handler import DARPAHandler
from .atlas_handler import ATLASHandler
from .darpa_handler5 import DARPAHandler5
from .optc_handler import OptcHandler
__all__ = ["DARPAHandler", "ATLASHandler"]

handler_map = {
    "theia": DARPAHandler,
    "cadets": DARPAHandler,
    "clearscope": DARPAHandler,
    "trace": DARPAHandler,
    "atlas": ATLASHandler,
    "cadets5": DARPAHandler5,
    "theia5": DARPAHandler5,
    "optc": OptcHandler
}

path_map = {
    "theia": "/mnt/bigdata/aptdata/data_files/theia",
    "cadets": "/mnt/bigdata/aptdata/data_files/cadets",
    "clearscope": "/mnt/bigdata/aptdata/data_files/clearscope",
    "trace": "/mnt/bigdata/aptdata/data_files/trace",
    "atlas": "/mnt/bigdata/aptdata/atlas_data",
    "cadets5": "/mnt/bigdata/aptdata/data_files5/cadets",
    "theia5": "/mnt/bigdata/aptdata/data_files5/theia",
    "optc": "/mnt/bigdata/aptdata/data_files_optc/day1"
}

def get_handler(name, train):
    cls = handler_map.get(name.lower())
    base_path = path_map.get(name)
    if base_path is None:
        raise ValueError(f"未配置数据路径: {name}")
    if cls is None:
        raise ValueError(f"未知数据集: {name}")
    return cls(base_path, train)
