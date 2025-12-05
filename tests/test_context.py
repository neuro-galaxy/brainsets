import os
import pytest

from brainsets.ds_wizard.context import CONTEXT_PATH
from brainsets.ds_wizard.context import (
    get_default_dataset_info,
    task_taxonomy,
    modalities,
    eeg_bids_specification,
)


def is_default_dataset_info_present(file_name: str):
    path = os.path.join(CONTEXT_PATH, file_name)
    return os.path.exists(path)


@pytest.mark.skipif(
    not is_default_dataset_info_present("default_dataset_info.csv"),
    reason="Default dataset info is not present",
)
def test_get_default_dataset_info():
    result = get_default_dataset_info("ds005555")
    assert len(result) > 0
    assert type(result) == list
    for item in result:
        assert "brainset" in item.keys()
        assert item["brainset"] == "ds005555"


@pytest.mark.skipif(
    not is_default_dataset_info_present("EEG_task_taxonomy.csv"),
    reason="Task taxonomy is not present",
)
def test_task_taxonomy():
    result = task_taxonomy()
    assert len(result) > 0
    assert type(result) == list
    for item in result:
        assert "Category" in item.keys()
        assert "Subcategory" in item.keys()
        assert "Description" in item.keys()


@pytest.mark.skipif(
    not is_default_dataset_info_present("modalities.csv"),
    reason="Modalities are not present",
)
def test_modalities():
    result = modalities()
    assert len(result) > 0
    assert type(result) == list
    for item in result:
        assert "Modality" in item.keys()
        assert "Description" in item.keys()


@pytest.mark.skipif(
    not is_default_dataset_info_present("eeg_bids_specs.md"),
    reason="EEG BIDS specification is not present",
)
def test_eeg_bids_specification():
    result = eeg_bids_specification()
    assert len(result) > 0
    assert type(result) == str
