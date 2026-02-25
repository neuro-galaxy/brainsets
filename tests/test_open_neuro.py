"""Tests for brainsets.utils.openneuro and related modules."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from brainsets.utils.bids_utils import parse_bids_filename
from brainsets.utils.openneuro import (
    fetch_all_filenames,
    fetch_eeg_recordings,
    fetch_participants_tsv,
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


class TestParseBidsFilename:
    def test_simple_pattern_eeg(self):
        result = parse_bids_filename("sub-01_task-Sleep_eeg.edf", "eeg")
        assert result is not None
        assert result["subject"] == "01"
        assert result["session"] is None
        assert result["task"] == "Sleep"
        assert result["acquisition"] is None
        assert result["run"] is None

    def test_full_pattern_eeg(self):
        result = parse_bids_filename(
            "sub-01_ses-01_task-Rest_acq-headband_run-02_eeg.vhdr", "eeg"
        )
        assert result is not None
        assert result["subject"] == "01"
        assert result["session"] == "01"
        assert result["task"] == "Rest"
        assert result["acquisition"] == "headband"
        assert result["run"] == "02"

    def test_with_path(self):
        result = parse_bids_filename("sub-01/eeg/sub-01_task-Sleep_eeg.edf", "eeg")
        assert result is not None
        assert result["subject"] == "01"
        assert result["task"] == "Sleep"

    def test_wrong_modality(self):
        result = parse_bids_filename("sub-01_task-Sleep_eeg.edf", "ieeg")
        assert result is None

    def test_non_bids_file(self):
        result = parse_bids_filename("sub-01_task-Sleep_bold.nii.gz", "eeg")
        assert result is None

    def test_invalid_pattern(self):
        result = parse_bids_filename("invalid_filename.edf", "eeg")
        assert result is None

    def test_ieeg_modality(self):
        result = parse_bids_filename("sub-01_task-VisualNaming_ieeg.edf", "ieeg")
        assert result is not None
        assert result["subject"] == "01"
        assert result["task"] == "VisualNaming"


@pytest.mark.integration
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


class TestFetchParticipantsTsv:
    @patch("brainsets.utils.openneuro.dataset.get_cached_s3_client")
    def test_fetch_participants_tsv_success(self, mock_get_client):
        tsv_content = "participant_id\tage\tsex\nsub-01\t25\tmale\nsub-02\t30\tfemale\n"
        mock_response = {"Body": MagicMock(read=lambda: tsv_content.encode("utf-8"))}

        mock_client = MagicMock()
        mock_client.get_object.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = fetch_participants_tsv("ds005555")

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "sub-01" in result.index
        assert "sub-02" in result.index
        assert result.loc["sub-01", "age"] == 25
        assert result.loc["sub-01", "sex"] == "male"
        assert result.loc["sub-02", "age"] == 30
        assert result.loc["sub-02", "sex"] == "female"

    @patch("brainsets.utils.openneuro.dataset.get_cached_s3_client")
    def test_fetch_participants_tsv_with_na_values(self, mock_get_client):
        tsv_content = "participant_id\tage\tsex\nsub-01\tn/a\tmale\nsub-02\t30\tn/a\n"
        mock_response = {"Body": MagicMock(read=lambda: tsv_content.encode("utf-8"))}

        mock_client = MagicMock()
        mock_client.get_object.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = fetch_participants_tsv("ds005555")

        assert result is not None
        assert pd.isna(result.loc["sub-01", "age"])
        assert result.loc["sub-01", "sex"] == "male"
        assert result.loc["sub-02", "age"] == 30
        assert pd.isna(result.loc["sub-02", "sex"])

    @patch("brainsets.utils.openneuro.dataset.get_cached_s3_client")
    def test_fetch_participants_tsv_file_not_found(self, mock_get_client):
        mock_client = MagicMock()
        error_response = {"Error": {"Code": "NoSuchKey"}}
        mock_client.get_object.side_effect = ClientError(error_response, "GetObject")
        mock_get_client.return_value = mock_client

        result = fetch_participants_tsv("ds005555")

        assert result is None

    @patch("brainsets.utils.openneuro.dataset.get_cached_s3_client")
    def test_fetch_participants_tsv_no_participant_id_column(self, mock_get_client):
        tsv_content = "subject\tage\tsex\nsub-01\t25\tmale\n"
        mock_response = {"Body": MagicMock(read=lambda: tsv_content.encode("utf-8"))}

        mock_client = MagicMock()
        mock_client.get_object.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = fetch_participants_tsv("ds005555")

        assert result is None

    @patch("brainsets.utils.openneuro.dataset.get_cached_s3_client")
    def test_fetch_participants_tsv_other_error(self, mock_get_client):
        mock_client = MagicMock()
        error_response = {"Error": {"Code": "AccessDenied"}}
        mock_client.get_object.side_effect = ClientError(error_response, "GetObject")
        mock_get_client.return_value = mock_client

        with pytest.raises(ClientError):
            fetch_participants_tsv("ds005555")
