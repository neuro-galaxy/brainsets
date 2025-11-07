import os
import time
from collections import defaultdict
from typing import Dict, Any
from pathlib import Path
import ray, ray.actor
from ray.util.actor_pool import ActorPool

from rich.live import Live
from rich.console import Console


@ray.remote
class StatusTracker:
    def __init__(self):
        self.statuses: Dict[Any, str] = {}

    def update_status(self, id: Any, status: str):
        self.statuses[id] = status

    def get_all_statuses(self):
        return self.statuses


class ProcessorBase:
    asset_id: str  # MUST be set by subclasses

    def __init__(self, tracker_handle: ray.actor.ActorHandle, raw_root, processed_root):
        self.tracker_handle = tracker_handle
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)

    def process_top(self, *args, **kwargs):
        try:
            self.download_and_process(*args, **kwargs)
            self.tracker_handle.update_status.remote(self.asset_id, "DONE")
        except:
            self.tracker_handle.update_status.remote(self.asset_id, "FAILED")

    def download_and_process(self, *args, **kwargs):
        raise NotImplementedError

    def update_status(self, status: str):
        self.tracker_handle.update_status.remote(self.asset_id, status)

    @classmethod
    def get_manifest(cls):
        raise NotImplementedError


def generate_status_table(status_dict: Dict[str, str]) -> str:
    # Sort files for a consistent display
    sorted_files = sorted(status_dict.keys())

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
        lambda actor, task: actor.process_top.remote(task),
        work_items,
    )
    for _ in results_generator:
        pass


def run_processor(processor_cls: ProcessorBase, raw_root, processed_root):
    pool_size = 5

    manifest_items = processor_cls.get_manifest()
    print(f"Discovered {len(manifest_items)} manifest items")

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"  # to avoid a warning
    ray.init("local", num_cpus=pool_size, log_to_driver=False)
    tracker = StatusTracker.remote()

    actors = [
        processor_cls.remote(tracker, raw_root, processed_root)
        for _ in range(pool_size)
    ]
    run_pool_in_background.remote(actors, manifest_items)

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
            if len(status_dict) == len(manifest_items):
                all_done = all(s in ["DONE", "FAILED"] for s in status_dict.values())

            # Sleep to prevent loop from spinning too fast
            time.sleep(0.1)

    console.print("\n[bold green]All processing pipelines complete![/bold green]")
