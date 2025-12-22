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
    """
    Pipeline for the Gonzales neurofeedback in virtual reality OpenNeuro dataset
    (ds006839, 2025 release).
    """

    parser = parser

    @classmethod
    def _get_identifiers(cls) -> tuple[str, str]:
        """
        Get brainset_id and dataset_id from config metadata.

        Returns
        -------
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

        super().process(download_output)
