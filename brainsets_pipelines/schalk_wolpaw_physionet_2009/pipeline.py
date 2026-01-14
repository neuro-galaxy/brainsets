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

from brainsets.descriptions import BrainsetDescription
from brainsets.taxonomy import Task
from brainsets.moabb_pipeline import MOABBPipeline


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

    def get_brainset_description(self):
        return BrainsetDescription(
            id="schalk_wolpaw_physionet_2009",
            origin_version="unknown",
            derived_version="1.0.0",
            source="https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html",
            description="PhysioNet Motor Imagery dataset: over 1500 EEG recordings "
            "from 109 volunteers performing motor imagery tasks.",
        )
