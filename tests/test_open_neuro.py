"""Tests for brainsets.utils.openneuro and related modules."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from brainsets.utils.bids_utils import parse_bids_eeg_filename
from brainsets.utils.mne_utils import rename_electrodes, set_channel_modalities
from brainsets.utils.openneuro import (
    fetch_all_filenames,
    fetch_eeg_recordings,
    validate_dataset_id,
)


class TestValidateDatasetId:
    def test_valid_formats(self):
        assert validate_dataset_id("ds006695") == "ds006695"
        assert validate_dataset_id("5085") == "ds005085"
        assert validate_dataset_id("ds5085") == "ds005085"
        assert validate_dataset_id("ds005085") == "ds005085"

    def test_invalid_formats(self):
        with pytest.raises(ValueError):
            validate_dataset_id("ds0050851")
        with pytest.raises(ValueError):
            validate_dataset_id("50851")
        with pytest.raises(ValueError):
            validate_dataset_id("ds0066951")
        with pytest.raises(ValueError):
            validate_dataset_id("invalid")


class TestParseBidsEegFilename:
    def test_simple_pattern(self):
        result = parse_bids_eeg_filename("sub-01_task-Sleep_eeg.edf")
        assert result is not None
        assert result["subject_id"] == "sub-01"
        assert result["session_id"] is None
        assert result["task_id"] == "Sleep"
        assert result["acq_id"] is None
        assert result["run_id"] is None

    def test_full_pattern(self):
        result = parse_bids_eeg_filename(
            "sub-01_ses-01_task-Rest_acq-headband_run-02_eeg.vhdr"
        )
        assert result is not None
        assert result["subject_id"] == "sub-01"
        assert result["session_id"] == "ses-01"
        assert result["task_id"] == "Rest"
        assert result["acq_id"] == "headband"
        assert result["run_id"] == "02"

    def test_with_path(self):
        result = parse_bids_eeg_filename("sub-01/eeg/sub-01_task-Sleep_eeg.edf")
        assert result is not None
        assert result["subject_id"] == "sub-01"
        assert result["task_id"] == "Sleep"

    def test_non_eeg_file(self):
        result = parse_bids_eeg_filename("sub-01_task-Sleep_bold.nii.gz")
        assert result is None

    def test_invalid_pattern(self):
        result = parse_bids_eeg_filename("invalid_filename.edf")
        assert result is None


class TestRenameElectrodes:
    def test_rename_channels(self):
        mock_raw = MagicMock()
        mock_raw.ch_names = ["PSG_F3", "PSG_F4", "C3"]

        rename_map = {"PSG_F3": "F3", "PSG_F4": "F4"}
        rename_electrodes(mock_raw, rename_map)

        mock_raw.rename_channels.assert_called_once_with(
            {"PSG_F3": "F3", "PSG_F4": "F4"}, allow_duplicates=False
        )

    def test_empty_rename_map(self):
        mock_raw = MagicMock()
        rename_electrodes(mock_raw, {})
        mock_raw.rename_channels.assert_not_called()

    def test_none_rename_map(self):
        mock_raw = MagicMock()
        rename_electrodes(mock_raw, None)
        mock_raw.rename_channels.assert_not_called()

    def test_partial_match(self):
        mock_raw = MagicMock()
        mock_raw.ch_names = ["F3", "F4"]

        rename_map = {"PSG_F3": "F3_new", "F4": "F4_new"}
        rename_electrodes(mock_raw, rename_map)

        mock_raw.rename_channels.assert_called_once_with(
            {"F4": "F4_new"}, allow_duplicates=False
        )


class TestSetChannelModalities:
    def test_set_modalities(self):
        mock_raw = MagicMock()
        mock_raw.ch_names = ["F3", "F4", "EOG_L", "EMG"]

        modality_map = {
            "EEG": ["F3", "F4"],
            "EOG": ["EOG_L"],
            "EMG": ["EMG"],
        }
        set_channel_modalities(mock_raw, modality_map)

        mock_raw.set_channel_types.assert_called_once()
        call_args = mock_raw.set_channel_types.call_args[0][0]
        assert call_args["F3"] == "eeg"
        assert call_args["F4"] == "eeg"
        assert call_args["EOG_L"] == "eog"
        assert call_args["EMG"] == "emg"

    def test_empty_modality_map(self):
        mock_raw = MagicMock()
        set_channel_modalities(mock_raw, {})
        mock_raw.set_channel_types.assert_not_called()

    def test_none_modality_map(self):
        mock_raw = MagicMock()
        set_channel_modalities(mock_raw, None)
        mock_raw.set_channel_types.assert_not_called()


@pytest.mark.parametrize("dataset_id,error", [("ds006695", False), ("ds00555", True)])
def test_fetch_all_filenames(dataset_id, error):
    if error:
        with pytest.raises(RuntimeError):
            fetch_all_filenames(dataset_id)
    else:
        filenames = fetch_all_filenames(dataset_id)
        assert filenames is not None
        assert isinstance(filenames, list)
        assert len(filenames) > 0

        has_nested_files = False
        for filename in filenames:
            assert isinstance(filename, str)
            assert len(filename) > 0
            assert not filename.endswith("/")
            assert not filename.startswith(dataset_id)
            if "/" in filename:
                has_nested_files = True

        assert has_nested_files


class TestFetchEegRecordings:
    @patch("brainsets.utils.openneuro.dataset.fetch_all_filenames")
    def test_fetch_eeg_recordings(self, mock_fetch_filenames):
        mock_fetch_filenames.return_value = [
            "sub-01/eeg/sub-01_task-Sleep_eeg.edf",
            "sub-01/eeg/sub-01_task-Sleep_channels.tsv",
            "sub-02/eeg/sub-02_ses-01_task-Rest_acq-headband_run-01_eeg.vhdr",
            "sub-02/eeg/sub-02_ses-01_task-Rest_acq-headband_run-01_eeg.vmrk",
            "participants.tsv",
        ]

        recordings = fetch_eeg_recordings("ds005555")

        assert len(recordings) == 2

        rec1 = next(r for r in recordings if r["subject_id"] == "sub-01")
        assert rec1["recording_id"] == "sub-01_task-Sleep"
        assert rec1["task_id"] == "Sleep"
        assert rec1["session_id"] is None

        rec2 = next(r for r in recordings if r["subject_id"] == "sub-02")
        assert rec2["recording_id"] == "sub-02_ses-01_task-Rest_acq-headband_run-01"
        assert rec2["task_id"] == "Rest"
        assert rec2["session_id"] == "ses-01"
        assert rec2["acq_id"] == "headband"
        assert rec2["run_id"] == "01"

    @patch("brainsets.utils.openneuro.dataset.fetch_all_filenames")
    def test_no_eeg_files(self, mock_fetch_filenames):
        mock_fetch_filenames.return_value = [
            "participants.tsv",
            "dataset_description.json",
        ]

        recordings = fetch_eeg_recordings("ds005555")
        assert len(recordings) == 0
