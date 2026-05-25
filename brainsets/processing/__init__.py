from . import signal
from .signal import downsample_wideband, extract_bands, cube_to_long

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "brainsets.processing.signal",
            "autosummary": [f"signal.{name}" for name in signal._functions],
        },
    ],
}
