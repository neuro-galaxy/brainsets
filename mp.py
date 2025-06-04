from typing import Union
from pathlib import Path

from brainsets.descriptions import BrainsetDescription
from brainsets import Brainset

from logging import getLogger

logger = getLogger(__name__)


class PerichMillerPopulation2019(Brainset):
    brainset_description = BrainsetDescription(
        id="perich_miller_population_2018",
        origin_version="dandi/000688/draft",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000688",
        description="This dataset contains electrophysiology and behavioral data from "
        "three macaques performing either a center-out task or a continuous random "
        "target acquisition task. Neural activity was recorded from "
        "chronically-implanted electrode arrays in the primary motor cortex (M1) or "
        "dorsal premotor cortex (PMd) of four rhesus macaque monkeys. A subset of "
        "sessions includes recordings from both regions simultaneously. The data "
        "contains spiking activity—manually spike sorted in three subjects, and "
        "threshold crossings in the fourth subject—obtained from up to 192 electrodes "
        "per session, cursor position and velocity, and other task related metadata.",
    )

    requirements = [
        "dandi==0.61.2",
        "scikit-learn==1.2.1",
        "brainsets @ git+https://github.com/neuro-galaxy/brainsets@main",
    ]

    def __init__(self, data_root: Union[str, Path]):
        super().__init__(data_root)

    def download_dataset(self, raw_dir: Path):
        cmd = f"dandi download -o {raw_dir} -e refresh DANDI:000140/0.220113.0408"
        print(f"executing {cmd}")
        self.shell_cmd(cmd)

        raw_file_list = list((raw_dir / "000140" / "sub-Jenkins").iterdir())
        return raw_file_list

    def process_recording(self, raw_file_path: Path): ...


print("yoooo")
# if __name__ == "__main__":
#     PerichMillerPopulation2019.process_payload_in_venv()
