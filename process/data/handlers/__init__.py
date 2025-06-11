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