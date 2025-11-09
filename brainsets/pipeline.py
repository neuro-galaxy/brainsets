import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
import os
import time
from collections import defaultdict
from typing import Dict, Any, Type, Optional
from pathlib import Path
import ray, ray.actor
from ray.util.actor_pool import ActorPool

from rich.pretty import pprint
from rich.live import Live
from rich.console import Console
import pandas as pd


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


@ray.remote
class StatusTracker:
    def __init__(self):
        self.statuses: Dict[Any, str] = {}

    def update_status(self, id: Any, status: str):
        self.statuses[id] = status

    def get_all_statuses(self):
        return self.statuses


def get_style(status):
    # Add color coding based on status
    if "DONE" in status:
        return "green"
    elif "FAILED" in status:
        return "red"
    elif "DOWNLOADING" in status:
        return "blue"
    else:
        return "yellow"


def generate_status_table(status_dict: Dict[str, str]) -> str:
    # Sort files for a consistent display
    sorted_files = sorted(status_dict.keys())

    ans = ""
    for file_id in sorted_files:
        status = status_dict[file_id]
        if status == "DONE":
            continue
        status_style = get_style(status)
        ans += f"{file_id}: [{status_style}]{status}[/]\n"

    summary_dict = defaultdict(int)
    for v in status_dict.values():
        summary_dict[v] += 1

    summary_list = [f"{v} [{get_style(k)}]{k}[/]" for k, v in summary_dict.items()]
    ans += "Summary: " + " | ".join(summary_list)
    return ans


@ray.remote
def run_pool_in_background(actors, work_items):
    r"""
    Run a pool of `actors` over `work_items` (a list of arguments)
    """
    pool = ActorPool(actors)
    results_generator = pool.map_unordered(
        lambda actor, task: actor.run_item.remote(task),
        work_items,
    )
    for _ in results_generator:
        pass


def run_parallel(
    pipeline_cls: Type[BrainsetPipeline],
    manifest: pd.DataFrame,
    raw_root: Path,
    processed_root: Path,
    num_jobs: int,
    extra_args: Namespace,
):
    actor_cls: ray.actor.ActorClass = ray.remote(pipeline_cls)

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"  # to avoid a warning
    ray.init("local", num_cpus=num_jobs, log_to_driver=False)
    tracker = StatusTracker.remote()

    actors = [
        actor_cls.remote(tracker, raw_root, processed_root, extra_args)
        for _ in range(num_jobs)
    ]
    run_pool_in_background.remote(actors, list(manifest.itertuples()))

    console = Console()
    with Live(
        generate_status_table({}),
        console=console,
        refresh_per_second=10,
    ) as live:
        all_done = False
        while not all_done:  # Spin loop
            # Get status and update TUI
            status_dict = ray.get(tracker.get_all_statuses.remote())
            live.update(generate_status_table(status_dict))

            # Check for completion
            if len(status_dict) == len(manifest):
                all_done = all(s in ["DONE", "FAILED"] for s in status_dict.values())

            # Sleep to prevent loop from spinning too fast
            time.sleep(0.1)

    console.print("\n[bold green]All processing pipelines complete![/bold green]")


def run(pipeline_cls: Type[BrainsetPipeline], args=None):
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("-s", "--single", default=None, type=str)
    parser.add_argument("-c", "--cores", default=4, type=int)
    args, remaining_args = parser.parse_known_args(args)

    pipeline_args = None
    if isinstance(pipeline_cls.parser, ArgumentParser):
        pipeline_args = pipeline_cls.parser.parse_args(remaining_args)

    raw_dir = args.raw_dir / pipeline_cls.brainset_id
    processed_dir = args.processed_dir / pipeline_cls.brainset_id

    manifest = pipeline_cls.get_manifest(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        args=pipeline_args,
    )
    print(f"Discovered {len(manifest)} manifest items")

    if args.single is None:
        run_parallel(
            pipeline_cls=pipeline_cls,
            manifest=manifest,
            raw_root=raw_dir,
            processed_root=processed_dir,
            num_jobs=args.cores,
            extra_args=pipeline_args,
        )
    else:
        manifest_item = manifest.loc[args.single]
        pipeline = pipeline_cls(
            tracker_handle=None,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            args=pipeline_args,
        )
        pipeline.process(pipeline.download(manifest_item))


def get_processor_from_pipeline_file(pipeline_filepath):
    # Load pipeline file as a module
    import importlib.util

    spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_filepath)
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    # return the Processor class
    return pipeline_module.Pipeline


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("pipeline_file", type=Path)
    parser.add_argument("--list", action="store_true")
    args, remaining_args = parser.parse_known_args()

    processor_cls = get_processor_from_pipeline_file(args.pipeline_file)

    if args.list:
        print(processor_cls.get_manifest())
        sys.exit(0)

    run(processor_cls, remaining_args)
