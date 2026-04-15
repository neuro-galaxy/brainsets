# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "pnpl~=0.0.8",
#   "temporaldata@git+https://github.com/neuro-galaxy/temporaldata@main",
# ]
# ///

from argparse import ArgumentParser
from pathlib import Path

from brainsets.pipeline import BrainsetPipeline


parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


class Pipeline(BrainsetPipeline):
    brainset_id = "libribrain_2025"
    origin_version = ""
    parser = parser
