from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
import os
import time
from collections import defaultdict
from typing import Dict, Any, Type
from pathlib import Path
import ray, ray.actor
from ray.util.actor_pool import ActorPool

from rich.pretty import pprint
from rich.live import Live
from rich.console import Console
import pandas as pd


@ray.remote
class StatusTracker:
    def __init__(self):
        self.statuses: Dict[Any, str] = {}

    def update_status(self, id: Any, status: str):
        self.statuses[id] = status

    def get_all_statuses(self):
        return self.statuses


class ProcessorBase(ABC):
    asset_id: str  # MUST be set by subclasses

    def __init__(
        self,
        tracker_handle: ray.actor.ActorHandle | None,
        raw_root: Path,
        processed_root: Path,
        args: Namespace | None,
    ):
        self.tracker_handle = tracker_handle
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.args = args

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> pd.DataFrame:
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

    @classmethod
    def parse_args(cls, arg_list) -> Namespace | None:
        pass

    @abstractmethod
    def process(self, args): ...

    def _main_(self, manifest_item):
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
        lambda actor, task: actor._main_.remote(task),
        work_items,
    )
    for _ in results_generator:
        pass


def run_parallel(
    processor_cls: Type[ProcessorBase],
    raw_root: Path,
    processed_root: Path,
    num_jobs: int,
    extra_args: Namespace,
):
    actor_cls: ray.actor.ActorClass = ray.remote(processor_cls)

    manifest = processor_cls.get_manifest()
    print(f"Discovered {len(manifest)} manifest items")

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


def run(processor_cls: Type[ProcessorBase], args=None):
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("-s", "--single", default=None, type=str)
    parser.add_argument("-c", "--cores", default=4, type=int)
    args, remaining_args = parser.parse_known_args(args)

    processor_args = processor_cls.parse_args(remaining_args)

    if args.single is None:
        run_parallel(
            processor_cls=processor_cls,
            raw_root=args.raw_dir,
            processed_root=args.processed_dir,
            num_jobs=args.cores,
            extra_args=processor_args,
        )
    else:
        manifest = processor_cls.get_manifest()
        manifest_item = manifest.loc[args.single]
        processor = processor_cls(
            tracker_handle=None,
            raw_root=args.raw_dir,
            processed_root=args.processed_dir,
            args=processor_args,
        )
        processor.process(processor.download(manifest_item))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("pipeline_file", type=Path)
    args, remaining_args = parser.parse_known_args()
    import importlib.util

    spec = importlib.util.spec_from_file_location("pipeline_module", args.pipeline_file)
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)

    run(pipeline_module.Processor, remaining_args)