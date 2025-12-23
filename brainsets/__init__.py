from importlib.metadata import version, PackageNotFoundError

from .core import serialize_fn_map

try:
    __version__ = version("brainsets")
except PackageNotFoundError:  # pragma: no cover
    # This can happen if someone is importing brainsets without installing
    pass  # pragma: no cover
