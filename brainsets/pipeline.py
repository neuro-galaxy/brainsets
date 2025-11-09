from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Optional
from pathlib import Path
import ray.actor
import pandas as pd
from rich.console import Console


class BrainsetPipeline(ABC):
    r"""Abstract base class for processing neural data into a standardized format.

    This class defines the interface for brainset pipelines, which handle the
    download and processing of neural datasets. Subclasses must implement the
    abstract methods to define how data is retrieved and transformed.

    The pipeline workflow consists of:
    1. Generating a manifest (list of assets to process) via `get_manifest()`
    2. Downloading each asset via `download()`
    3. Processing each downloaded asset via `process()`

    Attributes
    ----------
    Subclass-Defined Attributes
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    These attributes should be set by subclasses as class variables:

    brainset_id : str
        Unique identifier for the brainset. Must be set by the Pipeline subclass.
    parser : Optional[ArgumentParser]
        Optional argument parser for pipeline-specific command-line arguments.
        If set by a subclass, the runner will automatically parse any extra
        command-line arguments using this parser. The parsed arguments are then
        passed to `get_manifest()` as a method argument, and to the `download()` and
        `process()` methods via `self.args`.

    Runner-Managed Attributes
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    These attributes are automatically set by the runner and should not be
    modified by subclasses:

    asset_id : str
        Identifier for the current asset being processed. Set automatically in
        `run_item()`.
    tracker_handle : Optional[ray.actor.ActorHandle]
        Ray actor handle for distributed status tracking. If None, status updates
        are printed to console. Set by the runner during pipeline instantiation.
    raw_dir : Path
        Directory path for storing raw (downloaded) data. Set by the runner
        during pipeline instantiation.
    processed_dir : Path
        Directory path for storing processed (transformed) data. Set by the runner
        during pipeline instantiation.
    args : Optional[Namespace]
        Parsed command-line arguments for pipeline configuration. Contains the
        parsed arguments from `parser` if one is defined, otherwise None. Set by
        the runner during pipeline instantiation.

    Notes
    -----
    Subclasses must implement:
    - `get_manifest()`: Generate a DataFrame listing all assets to process
    - `download()`: Download a single asset from the manifest
    - `process()`: Transform downloaded data into standardized format

    Custom Command-Line Arguments
    ------------------------------
    Subclasses can define pipeline-specific command-line arguments by setting
    the class-level `parser` attribute to an `ArgumentParser` instance. The
    runner will automatically parse any extra arguments (after standard options
    like `--raw-dir`, `--processed-dir`, etc.) using this parser. The parsed
    arguments are passed to:
    - `get_manifest()` as the `args` parameter
    - The pipeline instance constructor as the `args` parameter
    - Accessible in instance methods via `self.args`

    Examples
    --------
    Subclasses should define the brainset_id and implement the abstract methods:

    >>> from argparse import ArgumentParser
    >>> parser = ArgumentParser()
    >>> parser.add_argument("--redownload", action="store_true")
    >>> parser.add_argument("--reprocess", action="store_true")
    >>>
    >>> class MyBrainsetPipeline(BrainsetPipeline):
    ...     brainset_id = "my_brainset"
    ...     parser = parser
    ...
    ...     @classmethod
    ...     def get_manifest(cls, raw_dir, processed_dir, args):
    ...         # Return DataFrame of assets to process
    ...         return pd.DataFrame(...)
    ...
    ...     def download(self, manifest_item):
    ...         # Download the asset
    ...         # return filepath or handle of downloaded data
    ...         ...
    ...
    ...     def process(self, download_output):
    ...         # Process the downloaded data
    ...         ...
    """

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
