import sys, os, shutil, inspect
from typing import Union, List
from pathlib import Path
import tempfile
import subprocess
import pickle

from brainsets.descriptions import BrainsetDescription

from logging import getLogger

logger = getLogger(__name__)


class Brainset:
    brainset_description: BrainsetDescription
    requirements: List[str]

    _tmpdir: Union[Path, None] = None
    _is_child_process = False

    def __init__(self, data_root: Union[str, Path]):
        self.data_dir = expand_path(data_root) / self.brainset_description.id

    def create_tmpdir(self) -> Path:
        self._tmpdir = Path(tempfile.mkdtemp())
        return self._tmpdir

    def cleanup_tmpdir(self):
        if self._tmpdir is not None:
            print(f"Deleting {self._tmpdir}")
            shutil.rmtree(self._tmpdir)
            self._tmpdir = None

    def create_isolated_environment(self) -> Path:
        if self._tmpdir is None:
            self.create_tmpdir()
            assert self._tmpdir is not None

        reqs_fpath = self._tmpdir / "requirements.txt"
        with open(reqs_fpath, "w") as f:
            f.write("\n".join(self.requirements))

        venv_path = self._tmpdir / "venv"
        venv_path.mkdir()
        print(f"Creating isolated environment at {venv_path}")
        print(f"Requirements file: {reqs_fpath}")
        subprocess.run(["uv", "venv", str(venv_path)], check=True)
        python_path = venv_path / "bin" / "python"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "-r",
                str(reqs_fpath),
                "--python",
                str(python_path),
            ],
            check=True,
        )

        return venv_path / "bin"

    def __del__(self):
        if not self._is_child_process:
            self.cleanup_tmpdir()

    def process(self, raw_root: Union[str, Path], n_cpus: int = 8):
        raw_dir = expand_path(raw_root) / self.brainset_description.id
        data_dir = self.data_dir

        raw_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        sys_path = self.create_isolated_environment()
        assert self._tmpdir is not None

        payload = {
            "__dict__": self.__dict__,
            "raw_dir": raw_dir,
            "n_cpus": n_cpus,
        }

        with open(self._tmpdir / "payload.pkl", "wb") as f:
            pickle.dump(payload, f)

        # Write launcher
        launcher_path = self._tmpdir / "run_in_venv.py"
        class_file = inspect.getfile(self.__class__)
        with open(launcher_path, "w") as f:
            f.write(
                f"""
import sys
sys.path.insert(0, '{os.path.dirname(class_file)}')
from {Path(class_file).stem} import {self.__class__.__name__}
{self.__class__.__name__}.process_payload_in_venv()
"""
            )

        subprocess.run([str(sys_path / "python"), str(launcher_path)], check=True)

    @classmethod
    def process_payload_in_venv(cls):
        obj = object.__new__(cls)
        obj._tmpdir = Path(sys.executable).parent.parent.parent
        obj._is_child_process = True

        # load payload
        with open(obj._tmpdir / "payload.pkl", "rb") as f:
            payload = pickle.load(f)

        for k, v in payload["__dict__"].items():
            setattr(obj, k, v)

        if hasattr(obj, "download_dataset"):
            raw_file_list = obj.download_dataset(raw_dir=payload["raw_dir"])
        else:
            raise ValueError("download_datasest not implemented")

        if payload["n_cpus"] == 1:
            for raw_file_path in raw_file_list:
                obj.process_recording(raw_file_path)

    def shell_cmd(self, cmd: str):
        assert self._tmpdir is not None
        sys_path = self._tmpdir / "venv" / "bin"
        cmd = f"PATH={str(sys_path)}:${{PATH}} {cmd}"
        os.system(cmd)


def expand_path(path: Union[str, Path]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path)))
