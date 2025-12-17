from argparse import ArgumentParser
from typing import Dict, Any

from brainsets.utils.open_neuro_pipeline import (
    OpenNeuroEEGPipeline,
    AutoLoadIdentifiersMeta,
)


parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


class Pipeline(OpenNeuroEEGPipeline, metaclass=AutoLoadIdentifiersMeta):
    """Pipeline for the Klinzing sleep OpenNeuro dataset (ds005555, 2024 release).

    This pipeline relies on the default implementations provided by
    :class:`OpenNeuroEEGPipeline` for:

    - :meth:`get_manifest()` – builds a recording-level manifest from OpenNeuro
    - :meth:`download()` – downloads EEG data for each recording from OpenNeuro
    - :meth:`process()` – loads EEG with MNE, extracts metadata and signal,
      and stores a standardized :class:`temporaldata.Data` object.

    Dataset-specific processing can be added later by overriding :meth:`process()`
    and calling ``super().process(download_output)``.

    The brainset_id and dataset_id are loaded from the config file.
    """

    parser = parser

    @classmethod
    def _get_identifiers(cls) -> tuple[str, str]:
        """
        Get brainset_id and dataset_id from config metadata.

        Returns
        -------
        tuple[str, str]
            Tuple of (brainset_id, dataset_id) extracted from config metadata.
        """
        config = cls._load_config(cls.config_file_path)
        metadata = config.get("dataset", {}).get("metadata", {})

        brainset_id = metadata.get("brainset_name")
        dataset_id = metadata.get("dataset_id")

        if brainset_id is None:
            raise KeyError(
                f"brainset_name not found in config metadata. "
                f"Config loaded from {cls.config_file_path}"
            )

        if dataset_id is None:
            raise KeyError(
                f"dataset_id not found in config metadata. "
                f"Config loaded from {cls.config_file_path}"
            )

        return brainset_id, dataset_id

    def process(self, download_output: Dict[str, Any]) -> None:
        """
        Process Klinzing sleep dataset (ds005555) downloads.
        """
        super().process(download_output)
