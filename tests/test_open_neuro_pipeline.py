"""Tests for brainsets.utils.open_neuro_pipeline module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from brainsets.utils.open_neuro_pipeline import OpenNeuroEEGPipeline


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
    @patch("brainsets.utils.open_neuro_pipeline.fetch_eeg_recordings")
    @patch("brainsets.utils.open_neuro_pipeline.construct_s3_url")
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

    @patch("brainsets.utils.open_neuro_pipeline.fetch_eeg_recordings")
    @patch("brainsets.utils.open_neuro_pipeline.construct_s3_url")
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

    @patch("brainsets.utils.open_neuro_pipeline.fetch_eeg_recordings")
    def test_get_manifest_no_recordings(self, mock_fetch_recordings):
        mock_fetch_recordings.return_value = []

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="No EEG recordings found"):
                MockPipeline.get_manifest(Path(temp_dir), None)


class TestApplyCommonMapping:
    def test_apply_mapping_with_both(self):
        pipeline = MockPipeline.__new__(MockPipeline)
        pipeline.ELECTRODE_RENAME = {"PSG_F3": "F3"}
        pipeline.MODALITY_CHANNELS = {"EEG": ["F3"]}
        pipeline.update_status = MagicMock()

        mock_raw = MagicMock()
        mock_raw.ch_names = ["PSG_F3", "C3"]

        with patch(
            "brainsets.utils.open_neuro_pipeline.rename_electrodes"
        ) as mock_rename:
            with patch(
                "brainsets.utils.open_neuro_pipeline.set_channel_modalities"
            ) as mock_modality:
                pipeline._apply_common_mapping(mock_raw)

        mock_rename.assert_called_once_with(mock_raw, {"PSG_F3": "F3"})
        mock_modality.assert_called_once_with(mock_raw, {"EEG": ["F3"]})

    def test_apply_mapping_without_mappings(self):
        pipeline = MockPipeline.__new__(MockPipeline)
        pipeline.ELECTRODE_RENAME = None
        pipeline.MODALITY_CHANNELS = None
        pipeline.update_status = MagicMock()

        mock_raw = MagicMock()

        with patch(
            "brainsets.utils.open_neuro_pipeline.rename_electrodes"
        ) as mock_rename:
            with patch(
                "brainsets.utils.open_neuro_pipeline.set_channel_modalities"
            ) as mock_modality:
                pipeline._apply_common_mapping(mock_raw)

        mock_rename.assert_not_called()
        mock_modality.assert_not_called()


class TestDownload:
    @patch("brainsets.utils.open_neuro_pipeline.check_recording_files_exist")
    @patch("brainsets.utils.open_neuro_pipeline.download_prefix_from_s3")
    def test_download_new_file(self, mock_download, mock_check_exists):
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
            manifest_item.s3_url = "s3://openneuro.org/ds005555/sub-01/eeg/sub-01_task-Sleep"

            result = pipeline.download(manifest_item)

            mock_download.assert_called_once()
            assert result["recording_id"] == "sub-01_task-Sleep"
            assert result["subject_id"] == "sub-01"

    @patch("brainsets.utils.open_neuro_pipeline.check_recording_files_exist")
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
            manifest_item.s3_url = "s3://openneuro.org/ds005555/sub-01/eeg/sub-01_task-Sleep"

            result = pipeline.download(manifest_item)

            assert result["recording_id"] == "sub-01_task-Sleep"
            pipeline.update_status.assert_any_call("Already Downloaded")

    @patch("brainsets.utils.open_neuro_pipeline.check_recording_files_exist")
    @patch("brainsets.utils.open_neuro_pipeline.download_prefix_from_s3")
    def test_download_with_redownload_flag(self, mock_download, mock_check_exists):
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
            manifest_item.s3_url = "s3://openneuro.org/ds005555/sub-01/eeg/sub-01_task-Sleep"

            result = pipeline.download(manifest_item)

            mock_download.assert_called_once()
            assert result["recording_id"] == "sub-01_task-Sleep"

    @patch("brainsets.utils.open_neuro_pipeline.check_recording_files_exist")
    @patch("brainsets.utils.open_neuro_pipeline.download_prefix_from_s3")
    def test_download_error_handling(self, mock_download, mock_check_exists):
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
            manifest_item.s3_url = "s3://openneuro.org/ds005555/sub-01/eeg/sub-01_task-Sleep"

            with pytest.raises(RuntimeError, match="Failed to download"):
                pipeline.download(manifest_item)
