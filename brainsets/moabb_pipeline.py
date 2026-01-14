"""Base pipeline class for MOABB (Mother of All BCI Benchmarks) datasets.

This module provides a reusable base class for integrating MOABB datasets into
the brainsets pipeline framework. Subclasses define the dataset and paradigm
classes, and implement paradigm-specific processing logic.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, Type
from pathlib import Path
import pandas as pd
import numpy as np

from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm
from moabb.utils import set_download_dir
from brainsets.pipeline import BrainsetPipeline


class MOABBPipeline(BrainsetPipeline):
    """Base class for MOABB dataset pipelines.

    Subclasses must define:
        - brainset_id: str
        - dataset_class: Type[BaseDataset]
        - paradigm_class: Type[BaseParadigm]
        - dataset_kwargs: Dict[str, Any] (optional, defaults to {})
        - paradigm_kwargs: Dict[str, Any] (optional, defaults to {})

    Subclasses must implement:
        - process(): Transform MOABB data to brainsets format

    The base class handles:
        - Manifest generation from dataset metadata
        - Data download via MOABB paradigm.get_data()
        - Session filtering
        - MNE download directory setup
    """

    dataset_class: Type[BaseDataset]
    paradigm_class: Type[BaseParadigm]
    dataset_kwargs: Dict[str, Any] = {}
    paradigm_kwargs: Dict[str, Any] = {}

    @classmethod
    def get_dataset(cls) -> BaseDataset:
        """Instantiate the MOABB dataset with configured kwargs."""
        return cls.dataset_class(**cls.dataset_kwargs)

    @classmethod
    def get_paradigm(cls) -> BaseParadigm:
        """Instantiate the MOABB paradigm with configured kwargs."""
        kwargs = {k: v for k, v in cls.paradigm_kwargs.items() if v is not None}
        return cls.paradigm_class(**kwargs)

    @classmethod
    def get_manifest(cls, raw_dir: Path, args) -> pd.DataFrame:
        """Auto-generate manifest from MOABB dataset metadata.

        Creates a manifest with one row per subject-session combination.
        The manifest index is 'session_id' formatted as 'subj-{subject:03d}_sess-{session}'.

        Parameters
        ----------
        raw_dir : Path
            Raw data directory (unused but required by base class)
        args : Namespace
            Pipeline arguments (unused but required by base class)

        Returns
        -------
        pd.DataFrame
            Manifest with columns: subject, session, session_id (as index)
        """
        dataset = cls.get_dataset()
        manifest_list = []

        for subject in dataset.subject_list:
            for session in range(dataset.n_sessions):
                session_id = f"subj-{subject:03d}_sess-{session}"
                manifest_list.append(
                    {
                        "subject": subject,
                        "session": session,
                        "session_id": session_id,
                    }
                )

        return pd.DataFrame(manifest_list).set_index("session_id")

    def download(self, manifest_item) -> Dict[str, Any]:
        """Download and extract data using MOABB paradigm.

        This method:
        1. Sets up MNE download directory
        2. Calls paradigm.get_data() to get epoched arrays
        3. Filters results to the specific session from manifest_item

        Parameters
        ----------
        manifest_item : NamedTuple
            Row from manifest containing subject and session info

        Returns
        -------
        dict
            Dictionary containing:
            - X: np.ndarray of shape (n_epochs, n_channels, n_samples)
            - labels: np.ndarray of shape (n_epochs,) with event names
            - meta: pd.DataFrame with columns: subject, session, run
            - info: MNE Info object from first epoch (for channel info)
        """
        self.update_status("DOWNLOADING")

        set_download_dir(str(self.raw_dir))

        dataset = self.get_dataset()
        paradigm = self.get_paradigm()

        X, labels, meta = paradigm.get_data(
            dataset=dataset,
            subjects=[manifest_item.subject.item()],
            return_epochs=False,
        )

        if len(X) == 0:
            raise ValueError(
                f"No epochs found for subject {manifest_item.subject}, "
                f"session {manifest_item.session}"
            )

        session_values = sorted(meta["session"].unique())
        if isinstance(manifest_item.session, int):
            if manifest_item.session < len(session_values):
                session_key = session_values[manifest_item.session]
            else:
                raise ValueError(
                    f"Session index {manifest_item.session} out of range for subject {manifest_item.subject}. "
                    f"Available {len(session_values)} sessions: {list(session_values)}"
                )
        else:
            session_key = str(manifest_item.session.item())
            if session_key not in session_values:
                raise ValueError(
                    f"Session {session_key} not found for subject {manifest_item.subject}. "
                    f"Available sessions: {list(session_values)}"
                )

        session_mask = meta["session"] == session_key
        if not session_mask.any():
            raise ValueError(
                f"No epochs found for subject {manifest_item.subject}, "
                f"session {session_key}"
            )

        X_filtered = X[session_mask]
        labels_filtered = labels[session_mask]
        meta_filtered = meta[session_mask].reset_index(drop=True)

        epochs, labels_epochs, meta_epochs = paradigm.get_data(
            dataset=dataset,
            subjects=[manifest_item.subject.item()],
            return_epochs=True,
        )

        session_mask_epochs = meta_epochs["session"] == session_key
        epochs_filtered = epochs[session_mask_epochs]

        info = epochs_filtered[0].info if len(epochs_filtered) > 0 else None

        return {
            "X": X_filtered,
            "labels": labels_filtered,
            "meta": meta_filtered,
            "info": info,
            "epochs": epochs_filtered,
        }

    @abstractmethod
    def process(self, download_output: Dict[str, Any]) -> None:
        """Transform MOABB data to brainsets format.

        Subclasses implement paradigm-specific processing:
        - Motor Imagery: extract trials with movement labels
        - P300: extract target/non-target epochs
        - SSVEP: extract frequency-tagged responses

        Parameters
        ----------
        download_output : dict
            Dictionary returned by download() containing X, labels, meta, info
        """
        ...
