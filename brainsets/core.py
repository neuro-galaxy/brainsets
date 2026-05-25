_functions = ["datetime_serialize_fn"]
_constants = ["serialize_fn_map"]

__all__ = _functions + _constants

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [
        {"title": "Functions", "autosummary": _functions},
        {"title": "Constants", "autosummary": _constants},
    ],
}

import datetime


def datetime_serialize_fn(obj, serialize_fn_map=None):
    r"""Convert a datetime object to a string."""
    return str(obj)


serialize_fn_map = {
    datetime.datetime: datetime_serialize_fn,
}
r"""A dict that maps classes to their serialization functions"""
