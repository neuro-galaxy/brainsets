from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Optional
from pathlib import Path
import ray.actor
import pandas as pd
from rich.console import Console


class BrainsetPipeline(ABC):
    brainset_id: str  # set by Pipeline subclass
    asset_id: str  # set in run_item()
    parser: Optional[ArgumentParser] = None

    def __init__(
        self,
        tracker_handle: Optional[ray.actor.ActorHandle],
        raw_dir: Path,
        processed_dir: Path,
        args: Optional[Namespace],
    ):
        self.tracker_handle = tracker_handle
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.args = args

    @classmethod
    @abstractmethod
    def get_manifest(
        cls,
        raw_dir: Path,
        processed_dir: Path,
        args: Optional[Namespace],
    ) -> pd.DataFrame:
        r"""Returns a dataframe, which is a table of assets to be download and processed.
        Each row will be passed individually to the `download` and `process` method.
        The index of this DataFrame will be used to identify assets for when user wants
        to process a single asset.
        """
        ...

    @abstractmethod
    def download(self, manifest_item):
        r"""
        Download the asset indicated by `manifest_item`. Return values
        """
        ...

    @abstractmethod
    def process(self, args): ...

    def run_item(self, manifest_item):
        self.asset_id = manifest_item.Index
        try:
            output = self.download(manifest_item)
            self.process(output)
            self.update_status("DONE")
        except:
            self.update_status("FAILED")

    def update_status(self, status: str):
        if self.tracker_handle is None:
            Console().print(f"[bold]Status:[/] [{get_style(status)}]{status}[/]")
        else:
            self.tracker_handle.update_status.remote(self.asset_id, status)
