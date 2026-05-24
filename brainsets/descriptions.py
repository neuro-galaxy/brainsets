_classes = [
    "BrainsetDescription",
    "SubjectDescription",
    "SessionDescription",
    "DeviceDescription",
]

__all__ = _classes

import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import temporaldata
from temporaldata import Data

import brainsets


class BrainsetDescription(Data):
    id: str
    origin_version: str
    derived_version: str
    source: str
    description: str
    brainsets_version: str
    temporaldata_version: str

    r"""A container for storing brainset metadata.

    Args:
        id: Unique identifier for the brainset
        origin_version: Version identifier for the original data source
        derived_version: Version identifier for the derived/processed data
        source: Original data source (usually a URL, or a short description otherwise)
        description: Text description of the brainset
        **kwargs: Any additional metadata
    """

    def __init__(
        self,
        id: str,
        origin_version: str,
        derived_version: str,
        source: str,
        description: str,
        **kwargs,
    ):
        super().__init__(
            id=id,
            origin_version=origin_version,
            derived_version=derived_version,
            source=source,
            description=description,
            brainsets_version=brainsets.__version__,
            temporaldata_version=temporaldata.__version__,
            **kwargs,
        )


class SubjectDescription(Data):
    r"""A container for storing subject related metadata.

    Fields are automatically normalized during construction:
    - ``species`` accepts a Species enum, string, int, or None (defaults to Species.UNKNOWN)
    - ``age`` accepts a float, int, numeric string, or None (defaults to 0.0)
    - ``sex`` accepts a Sex enum, string, int, or None (defaults to Sex.UNKNOWN)

    Args:
        id: Unique identifier for the subject
        species: Species of the subject, defaults to None
        age: Age of the subject (in days).
            It will be converted to float if not None. defaults to None
        sex: Sex of the subject, deafults to None
    """

    id: str
    species: str
    age: float | None
    sex: str | None

    def __init__(
        self,
        id: str,
        species: str | None = None,
        age: int | str | float | None = None,
        sex: str | None = None,
        **kwargs,
    ):

        if age is not None:
            age = self._normalize_age(age)

        super().__init__(
            id=id,
            species=species,
            age=age,
            sex=sex,
            **kwargs,
        )

    @classmethod
    def _normalize_age(cls, age) -> float | None:
        """Normalize and validate age value to a float in days."""

        if isinstance(age, (int, float)):
            age_normalized = float(age)
            if age_normalized < 0:
                raise ValueError(f"Age cannot be negative, got {age_normalized}")
            return age_normalized

        if isinstance(age, str):
            try:
                age_normalized = float(age)
            except (ValueError, TypeError):
                return 0.0
            else:
                if age_normalized < 0:
                    raise ValueError(f"Age cannot be negative, got {age_normalized}")
                return age_normalized

        raise TypeError(
            f"Age must be a float, int, numeric string, or None, got {type(age).__name__}"
        )


class SessionDescription(Data):
    r"""A container to store experimental session related metadata.

    Args:
        id: Unique identifier for the session
        recording_date: Date and time when the recording was made, defaults to None
        **kwargs: Any additional metadata
    """

    id: str
    recording_date: datetime.datetime | None

    def __init__(
        self,
        id: str,
        recording_date: datetime.datetime | None = None,
        **kwargs,
    ):
        super().__init__(
            id=id,
            recording_date=recording_date,
            **kwargs,
        )


class DeviceDescription(Data):
    r"""A container for storing recording device metadata.

    Args:
        id: Identifier for the device
        **kwargs: Any additional metadata
    """

    id: str

    def __init__(self, id: str, **kwargs: Any):
        super().__init__(id=id, **kwargs)
