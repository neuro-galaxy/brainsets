# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "moabb==1.4.3",
#   "scikit-learn==1.8.0",
# ]
# ///

"""Pipeline for PhysionetMI Motor Imagery dataset using MOABB.

This pipeline downloads and processes EEG motor imagery data from the PhysioNet
dataset using the MOABB dataset loader. The dataset consists of over 1500 one-
and two-minute EEG recordings obtained from 109 volunteers performing motor
imagery tasks.
"""

from argparse import ArgumentParser
import logging

from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

from temporaldata import Data
from brainsets.descriptions import BrainsetDescription
from brainsets.taxonomy import Task
from brainsets.utils.moabb.pipeline import MOABBPipeline
from brainsets.utils.split import (
    generate_trial_folds_by_task,
    generate_subject_kfold_assignment,
)


logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


class Pipeline(MOABBPipeline):
    brainset_id = "schalk_wolpaw_physionet_2009"
    parser = parser

    dataset_class = PhysionetMI
    paradigm_class = MotorImagery
    dataset_kwargs = {"imagined": True, "executed": False}

    task = Task.MOTOR_IMAGERY
    trial_key = "motor_imagery_trials"
    label_field = "movements"
    id_field = "movement_ids"
    stratify_field = "movements"
    label_map = {
        "left_hand": 0,
        "right_hand": 1,
        "hands": 2,
        "feet": 3,
        "rest": 4,
    }

    TASK_CONFIGS = {
        "MotorImagery_all": ["left_hand", "right_hand", "hands", "feet", "rest"],
        "MotorImagery_norest": ["left_hand", "right_hand", "hands", "feet"],
        "LeftRightImagery": ["left_hand", "right_hand"],
        "RightHandFeetImagery": ["right_hand", "feet"],
    }

    def _generate_splits(self, trials, subject_id: str = None):
        """Generate task-specific and subject-level k-fold splits.

        Generates:
        1. Task-specific within-session stratified k-fold splits for MotorImagery,
           LeftRightImagery, and RightHandFeetImagery tasks
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
        task_splits = generate_trial_folds_by_task(
            trials,
            task_configs=self.TASK_CONFIGS,
            label_field=self.label_field,
            n_folds=3,
            val_ratio=0.2,
            seed=42,
        )

        if subject_id is not None:
            subject_assignments = generate_subject_kfold_assignment(
                subject_id, n_folds=3, val_ratio=0.2, seed=42
            )
            return Data(**task_splits, **subject_assignments, domain=trials)
        else:
            return Data(**task_splits, domain=trials)

    def get_brainset_description(self):
        return BrainsetDescription(
            id="schalk_wolpaw_physionet_2009",
            origin_version="unknown",
            derived_version="1.0.0",
            source="https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html",
            description="PhysioNet Motor Imagery dataset: over 1500 EEG recordings "
            "from 109 volunteers performing motor imagery tasks.",
        )
