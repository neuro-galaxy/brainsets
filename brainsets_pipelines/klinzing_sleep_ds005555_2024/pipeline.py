from argparse import ArgumentParser
from typing import Dict, Any

from brainsets.utils.open_neuro_pipeline import OpenNeuroEEGPipeline


# Define CLI arguments specific to this pipeline
parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


class Pipeline(OpenNeuroEEGPipeline):
    """Pipeline for the Klinzing sleep OpenNeuro dataset (ds005555, 2024 release).

    This pipeline relies on the default implementations provided by
    :class:`OpenNeuroEEGPipeline` for:

    - :meth:`get_manifest()` – builds a subject-level manifest from OpenNeuro
    - :meth:`download()` – downloads EEG data for each subject from OpenNeuro
    - :meth:`process()` – loads EEG with MNE, extracts metadata and signal,
      and stores a standardized :class:`temporaldata.Data` object.

    Dataset-specific processing can be added later by overriding :meth:`process()`
    and calling ``super().process(download_output)``.
    """

    # Required identifiers
    brainset_id = "klinzing_sleep_ds005555_2024"
    dataset_id = "ds005555"
    parser = parser

    # Implement abstract configuration hooks required by OpenNeuroEEGPipeline
    @classmethod
    def get_brainset_id(cls) -> str:
        return cls.brainset_id

    @classmethod
    def get_dataset_id(cls) -> str:
        return cls.dataset_id

    def process(self, download_output: Dict[str, Any]) -> None:
        """Process Klinzing sleep dataset (ds005555) downloads.

        This implementation delegates the core EEG processing to
        :class:`OpenNeuroEEGPipeline`, then leaves a hook for adding
        dataset-specific post-processing steps.
        """
        # Run the default OpenNeuro EEG processing:
        #  - finds EEG files in the downloaded subject directory
        #  - loads them with MNE
        #  - extracts metadata, EEG signal and channels
        #  - writes standardized HDF5 files in ``self.processed_dir``
        super().process(download_output)

        # ------------------------------------------------------------------
        # Dataset-specific post-processing can be added here.
        # ------------------------------------------------------------------
