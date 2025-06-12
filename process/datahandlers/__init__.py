from .darpa_handler import DARPAHandler
from .atlas_handler import ATLASHandler

__all__ = ["DARPAHandler", "ATLASHandler"]

handler_map = {
    "darpa": DARPAHandler,
    "atlas": ATLASHandler}

path_map = {
    "darpa": "../data_files/theia",
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
