# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "moabb==1.4.3",
#   "scikit-learn==1.8.0",
# ]
# ///

"""Pipeline for BI2014a P300 dataset using MOABB.

This pipeline downloads and processes EEG P300 data from the Brain Invaders 2014a
dataset using the MOABB dataset loader. The dataset consists of EEG recordings
from 71 subjects performing a visual P300 Brain-Computer Interface task using
16 active dry electrodes across up to 3 sessions.
"""

from argparse import ArgumentParser
import logging

from moabb.datasets import BI2014a
from moabb.paradigms import P300

from temporaldata import Data
from brainsets.descriptions import BrainsetDescription
from brainsets.taxonomy import Task
from brainsets.moabb_pipeline import MOABBPipeline
from brainsets.utils.split import compute_subject_kfold_assignments


logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


class Pipeline(MOABBPipeline):
    brainset_id = "korczowski_brain_invaders_2014a"
    parser = parser

    dataset_class = BI2014a
    paradigm_class = P300
    dataset_kwargs = {}

    task = Task.P300
    trial_key = "p300_trials"
    label_field = "targets"
    id_field = "target_ids"
    stratify_field = "targets"
    label_map = {
        "Target": 1,
        "NonTarget": 0,
    }

    def _generate_splits(self, trials, subject_id: str = None):
        """Generate stratified folds and subject-level k-fold assignments.

        Generates:
        1. Standard stratified k-fold splits (from parent class)
        2. Subject-level k-fold assignments (train/valid/test per fold)

        Parameters
        ----------
        trials : Interval
            Trial intervals with label fields
        subject_id : str
            Subject identifier for subject-level splits

        Returns
        -------
        splits : Data
            Data object containing all split masks
        """
        splits = super()._generate_splits(trials, subject_id=subject_id)

        if subject_id is not None:
            subject_assignments = compute_subject_kfold_assignments(
                subject_id, n_folds=5, val_ratio=0.2, seed=42
            )
            for key, value in subject_assignments.items():
                setattr(splits, key, value)

        return splits

    def get_brainset_description(self):
        return BrainsetDescription(
            id="korczowski_brain_invaders_2014a",
            origin_version="unknown",
            derived_version="1.0.0",
            source="https://moabb.neurotechx.com/docs/generated/moabb.datasets.BI2014a.html",
            description="Brain Invaders 2014a P300 dataset: EEG recordings from "
            "71 subjects performing a visual P300 Brain-Computer Interface task "
            "using 16 active dry electrodes.",
        )
