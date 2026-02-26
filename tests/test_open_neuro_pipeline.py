"""Tests for brainsets.utils.openneuro.pipeline module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from temporaldata import Interval

from brainsets.utils.openneuro import OpenNeuroEEGPipeline


class MockPipeline(OpenNeuroEEGPipeline):
    """Mock pipeline for testing."""

    brainset_id = "test_brainset"
    dataset_id = "ds005555"

    ELECTRODE_RENAME = {
        "PSG_F3": "F3",
        "PSG_F4": "F4",
    }

    MODALITY_CHANNELS = {
        "EEG": ["F3", "F4"],
        "EOG": ["EOG_L"],
    }

    def process(self, download_output):
        super().process(download_output)


class TestGetManifest:
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    @patch("brainsets.utils.openneuro.pipeline.construct_s3_url_from_path")
    def test_get_manifest_dynamic(self, mock_construct_url, mock_fetch_recordings):
        mock_fetch_recordings.return_value = [
            {
                "recording_id": "sub-01_task-Sleep",
                "subject_id": "sub-01",
                "session_id": None,
                "task_id": "Sleep",
                "acq_id": None,
                "run_id": None,
                "eeg_file": "sub-01/eeg/sub-01_task-Sleep_eeg.edf",
            },
            {
                "recording_id": "sub-02_task-Sleep",
                "subject_id": "sub-02",
                "session_id": None,
                "task_id": "Sleep",
                "acq_id": None,
                "run_id": None,
                "eeg_file": "sub-02/eeg/sub-02_task-Sleep_eeg.edf",
            },
        ]
        mock_construct_url.side_effect = [
            "s3://openneuro.org/ds005555/sub-01/eeg/sub-01_task-Sleep",
            "s3://openneuro.org/ds005555/sub-02/eeg/sub-02_task-Sleep",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = MockPipeline.get_manifest(Path(temp_dir), None)

        assert len(manifest) == 2
        assert "sub-01_task-Sleep" in manifest.index
        assert "sub-02_task-Sleep" in manifest.index
        assert manifest.loc["sub-01_task-Sleep", "subject_id"] == "sub-01"

    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    @patch("brainsets.utils.openneuro.pipeline.construct_s3_url_from_path")
    def test_get_manifest_with_subject_filter(
        self, mock_construct_url, mock_fetch_recordings
    ):
        mock_fetch_recordings.return_value = [
            {
                "recording_id": "sub-01_task-Sleep",
                "subject_id": "sub-01",
                "session_id": None,
                "task_id": "Sleep",
                "acq_id": None,
                "run_id": None,
                "eeg_file": "sub-01/eeg/sub-01_task-Sleep_eeg.edf",
            },
            {
                "recording_id": "sub-02_task-Sleep",
                "subject_id": "sub-02",
                "session_id": None,
                "task_id": "Sleep",
                "acq_id": None,
                "run_id": None,
                "eeg_file": "sub-02/eeg/sub-02_task-Sleep_eeg.edf",
            },
        ]
        mock_construct_url.return_value = (
            "s3://openneuro.org/ds005555/sub-01/eeg/sub-01_task-Sleep"
        )

        class FilteredPipeline(MockPipeline):
            subject_ids = ["sub-01"]

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = FilteredPipeline.get_manifest(Path(temp_dir), None)

        assert len(manifest) == 1
        assert "sub-01_task-Sleep" in manifest.index

    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_no_recordings(self, mock_fetch_recordings):
        mock_fetch_recordings.return_value = []

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="No EEG recordings found"):
                MockPipeline.get_manifest(Path(temp_dir), None)


class TestApplyChannelMapping:
    def test_apply_mapping_with_both(self):
        pipeline = MockPipeline.__new__(MockPipeline)
        pipeline.ELECTRODE_RENAME = {"PSG_F3": "F3"}
        pipeline.MODALITY_CHANNELS = {"EEG": ["F3"]}
        pipeline.update_status = MagicMock()

        raw = MagicMock()
        raw.ch_names = ["PSG_F3", "C3"]
        raw.get_channel_types.return_value = ["misc", "misc"]

        result = pipeline._build_channels(raw, "test_recording", None)

        assert result.id[0] == "F3"
        assert result.id[1] == "C3"
        assert result.types[0] == "EEG"

    def test_apply_mapping_without_mappings(self):
        pipeline = MockPipeline.__new__(MockPipeline)
        pipeline.ELECTRODE_RENAME = None
        pipeline.MODALITY_CHANNELS = None
        pipeline.update_status = MagicMock()

        raw = MagicMock()
        raw.ch_names = ["F3", "C3"]
        raw.get_channel_types.return_value = ["misc", "misc"]

        result = pipeline._build_channels(raw, "test_recording", None)

        assert list(result.id) == ["F3", "C3"]
        assert list(result.types) == ["misc", "misc"]


class TestGenerateSplits:
    def test_generate_splits_creates_expected_structure(self):
        pipeline = MockPipeline.__new__(MockPipeline)
        domain = Interval(start=np.array([0.0]), end=np.array([100.0]))

        splits = pipeline._generate_splits(
            domain=domain, subject_id="sub-01", session_id="ses-01"
        )

        assert np.isclose(splits.train.start[0], 0.0)
        assert np.isclose(splits.train.end[0], 90.0)
        assert np.isclose(splits.valid.start[0], 90.0)
        assert np.isclose(splits.valid.end[0], 100.0)
        assert not hasattr(splits, "test")
        assert splits.intersubject_assignment in {"train", "valid"}
        assert splits.intersession_assignment in {"train", "valid"}

    def test_generate_splits_respects_custom_split_ratios(self):
        pipeline = MockPipeline.__new__(MockPipeline)
        pipeline.split_ratios = (0.8, 0.2)
        domain = Interval(start=np.array([0.0]), end=np.array([100.0]))

        splits = pipeline._generate_splits(
            domain=domain, subject_id="sub-01", session_id="ses-01"
        )

        assert np.isclose(splits.train.end[0], 80.0)
        assert np.isclose(splits.valid.end[0], 100.0)

    @patch("brainsets.utils.openneuro.pipeline.generate_string_kfold_assignment")
    def test_generate_splits_uses_matching_assignment_ratio(
        self, mock_generate_assignment
    ):
        pipeline = MockPipeline.__new__(MockPipeline)
        pipeline.split_ratios = (0.9, 0.1)
        domain = Interval(start=np.array([0.0]), end=np.array([100.0]))
        mock_generate_assignment.return_value = ["test"] + ["train"] * 9

        splits = pipeline._generate_splits(
            domain=domain, subject_id="sub-01", session_id="ses-01"
        )

        assert mock_generate_assignment.call_count == 2
        for call in mock_generate_assignment.call_args_list:
            assert call.kwargs["n_folds"] == 10
            assert call.kwargs["val_ratio"] == 0.0
            assert call.kwargs["seed"] == 42
        assert splits.intersubject_assignment == "valid"
        assert splits.intersession_assignment == "valid"


class TestDownload:
    @patch("brainsets.utils.openneuro.pipeline.download_dataset_description")
    @patch("brainsets.utils.openneuro.pipeline.check_recording_files_exist")
    @patch("brainsets.utils.openneuro.pipeline.download_recording")
    def test_download_new_file(
        self, mock_download, mock_check_exists, mock_download_dataset_description
    ):
        mock_check_exists.return_value = False

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"

            pipeline = MockPipeline(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                args=None,
            )
            pipeline.update_status = MagicMock()

            manifest_item = MagicMock()
            manifest_item.Index = "sub-01_task-Sleep"
            manifest_item.subject_id = "sub-01"
            manifest_item.session_id = None
            manifest_item.s3_url = (
                "s3://openneuro.org/ds005555/sub-01/eeg/sub-01_task-Sleep"
            )

            result = pipeline.download(manifest_item)

            mock_download.assert_called_once()
            mock_download_dataset_description.assert_called_once()
            assert result["recording_id"] == "sub-01_task-Sleep"
            assert result["subject_id"] == "sub-01"

    @patch("brainsets.utils.openneuro.pipeline.check_recording_files_exist")
    def test_download_file_exists(self, mock_check_exists):
        mock_check_exists.return_value = True

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"

            pipeline = MockPipeline(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                args=None,
            )
            pipeline.update_status = MagicMock()

            manifest_item = MagicMock()
            manifest_item.Index = "sub-01_task-Sleep"
            manifest_item.subject_id = "sub-01"
            manifest_item.session_id = None
            manifest_item.s3_url = (
                "s3://openneuro.org/ds005555/sub-01/eeg/sub-01_task-Sleep"
            )

            result = pipeline.download(manifest_item)

            assert result["recording_id"] == "sub-01_task-Sleep"
            pipeline.update_status.assert_any_call("Already Downloaded")

    @patch("brainsets.utils.openneuro.pipeline.download_dataset_description")
    @patch("brainsets.utils.openneuro.pipeline.check_recording_files_exist")
    @patch("brainsets.utils.openneuro.pipeline.download_recording")
    def test_download_with_redownload_flag(
        self, mock_download, mock_check_exists, mock_download_dataset_description
    ):
        mock_check_exists.return_value = True

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"

            args = MagicMock()
            args.redownload = True

            pipeline = MockPipeline(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                args=args,
            )
            pipeline.update_status = MagicMock()

            manifest_item = MagicMock()
            manifest_item.Index = "sub-01_task-Sleep"
            manifest_item.subject_id = "sub-01"
            manifest_item.session_id = None
            manifest_item.s3_url = (
                "s3://openneuro.org/ds005555/sub-01/eeg/sub-01_task-Sleep"
            )

            result = pipeline.download(manifest_item)

            mock_download.assert_called_once()
            mock_download_dataset_description.assert_called_once()
            assert result["recording_id"] == "sub-01_task-Sleep"

    @patch("brainsets.utils.openneuro.pipeline.download_dataset_description")
    @patch("brainsets.utils.openneuro.pipeline.check_recording_files_exist")
    @patch("brainsets.utils.openneuro.pipeline.download_recording")
    def test_download_error_handling(
        self, mock_download, mock_check_exists, mock_download_dataset_description
    ):
        mock_check_exists.return_value = False
        mock_download.side_effect = RuntimeError("Download failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"

            pipeline = MockPipeline(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                args=None,
            )
            pipeline.update_status = MagicMock()

            manifest_item = MagicMock()
            manifest_item.Index = "sub-01_task-Sleep"
            manifest_item.subject_id = "sub-01"
            manifest_item.session_id = None
            manifest_item.s3_url = (
                "s3://openneuro.org/ds005555/sub-01/eeg/sub-01_task-Sleep"
            )

            with pytest.raises(RuntimeError, match="Failed to download"):
                pipeline.download(manifest_item)
            mock_download_dataset_description.assert_not_called()


class TestGetSubjectInfo:
    @patch("brainsets.utils.openneuro.pipeline.fetch_participants_tsv")
    def test_get_subject_info_with_data(self, mock_fetch_participants):
        participants_df = pd.DataFrame(
            {"age": [25, 30], "sex": ["male", "female"]},
            index=["sub-01", "sub-02"],
        )
        participants_df.index.name = "participant_id"
        mock_fetch_participants.return_value = participants_df

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"

            pipeline = MockPipeline(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                args=None,
            )

            result = pipeline.get_subject_info("sub-01")

            assert result["age"] == 25
            assert result["sex"] == "male"

    @patch("brainsets.utils.openneuro.pipeline.fetch_participants_tsv")
    def test_get_subject_info_subject_not_found(self, mock_fetch_participants):
        participants_df = pd.DataFrame(
            {"age": [25], "sex": ["male"]},
            index=["sub-01"],
        )
        participants_df.index.name = "participant_id"
        mock_fetch_participants.return_value = participants_df

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"

            pipeline = MockPipeline(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                args=None,
            )

            result = pipeline.get_subject_info("sub-99")

            assert result["age"] is None
            assert result["sex"] is None

    @patch("brainsets.utils.openneuro.pipeline.fetch_participants_tsv")
    def test_get_subject_info_no_participants_file(self, mock_fetch_participants):
        mock_fetch_participants.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"

            pipeline = MockPipeline(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                args=None,
            )

            result = pipeline.get_subject_info("sub-01")

            assert result["age"] is None
            assert result["sex"] is None

    @patch("brainsets.utils.openneuro.pipeline.fetch_participants_tsv")
    def test_get_subject_info_with_na_values(self, mock_fetch_participants):
        participants_df = pd.DataFrame(
            {"age": [25, pd.NA], "sex": ["male", pd.NA]},
            index=["sub-01", "sub-02"],
        )
        participants_df.index.name = "participant_id"
        mock_fetch_participants.return_value = participants_df

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"

            pipeline = MockPipeline(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                args=None,
            )

            result = pipeline.get_subject_info("sub-02")

            assert result["age"] is None
            assert result["sex"] is None

    @patch("brainsets.utils.openneuro.pipeline.fetch_participants_tsv")
    def test_get_subject_info_cached(self, mock_fetch_participants):
        participants_df = pd.DataFrame(
            {"age": [25], "sex": ["male"]},
            index=["sub-01"],
        )
        participants_df.index.name = "participant_id"
        mock_fetch_participants.return_value = participants_df

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"

            pipeline = MockPipeline(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                args=None,
            )

            pipeline.get_subject_info("sub-01")
            pipeline.get_subject_info("sub-01")

            mock_fetch_participants.assert_called_once()

    def test_get_subject_info_override(self):
        class CustomPipeline(MockPipeline):
            def get_subject_info(self, subject_id: str) -> dict:
                return {"age": 99, "sex": "other"}

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            processed_dir = Path(temp_dir) / "processed"

            pipeline = CustomPipeline(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                args=None,
            )

            result = pipeline.get_subject_info("sub-01")

            assert result["age"] == 99
            assert result["sex"] == "other"
