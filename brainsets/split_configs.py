from dataclasses import dataclass
from typing import Literal, Protocol

from temporaldata import Data, Interval


class BaseSplitConfig(Protocol):
    """Contract for all split configurations."""

    def resolve(self, data: Data) -> Interval:
        """Return the sampling interval for this split."""
        ...

    @property
    def partition(self) -> str:
        """'train', 'valid', or 'test'"""
        ...


@dataclass
class KempSleepSplitConfig:
    """Split configuration for kemp_sleep_edf_2013.

    Uses 3-fold stratified cross-validation based on sleep stage labels.
    """

    fold: Literal[0, 1, 2]
    partition: Literal["train", "valid", "test"]

    def __post_init__(self):
        if self.fold not in (0, 1, 2):
            raise ValueError(f"fold must be 0, 1, or 2, got {self.fold}")
        if self.partition not in ("train", "valid", "test"):
            raise ValueError(f"partition must be 'train', 'valid', or 'test'")

    def resolve(self, data: Data) -> Interval:
        fold_data = getattr(data.splits.intrasubject, f"fold_{self.fold}")
        return getattr(fold_data, self.partition)

    @classmethod
    def available_folds(cls) -> list[int]:
        return [0, 1, 2]
