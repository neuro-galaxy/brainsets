import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from argparse import Namespace
import pandas as pd

from brainsets.utils.open_neuro_pipeline import (
    OpenNeuroEEGPipeline,
    AutoLoadIdentifiersMeta,
)


class TestPipeline(OpenNeuroEEGPipeline, metaclass=AutoLoadIdentifiersMeta):
    """Concrete implementation of OpenNeuroEEGPipeline for testing."""

    @classmethod
    def _get_config_file_path(cls) -> Path:
        return Path("/tmp/test_config.json")

    @classmethod
    def _get_identifiers(cls) -> tuple[str, str]:
        return "test_brainset", "ds006695"


class TestLoadConfig:

    def test_load_config_valid_json(self):
        """Test loading a valid JSON config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "dataset": {
                    "metadata": {
                        "name": "Test Dataset",
                        "version": "1.0.0",
                        "brainset_name": "test_brainset",
                        "dataset_id": "ds006695",
                    }
                }
            }
            json.dump(config_data, f)
            config_path = Path(f.name)

        try:
            result = TestPipeline._load_config(config_path)
            assert isinstance(result, dict)
            assert result["dataset"]["metadata"]["name"] == "Test Dataset"
            assert result["dataset"]["metadata"]["version"] == "1.0.0"
        finally:
            config_path.unlink()

    def test_load_config_invalid_json(self):
        """Test loading an invalid JSON config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content {")
            config_path = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                TestPipeline._load_config(config_path)
        finally:
            config_path.unlink()


class TestGetManifest:
    @pytest.fixture
    def mock_config(self):
        return {
            "dataset": {
                "metadata": {
                    "name": "Test Dataset",
                    "version": "1.0.0",
                },
                "channel_maps": {
                    "channel_maps": {
                        "map1": {
                            "channels": {
                                "Fp1": {"new_name": "Fp1", "modality": "EEG"},
                                "Fp2": {"new_name": "Fp2", "modality": "EEG"},
                            },
                            "device_name": "Test Device",
                            "device_manufacturer": "Test Manufacturer",
                        }
                    }
                },
                "recording_info": {
                    "recording_info": [
                        {
                            "recording_id": "sub-1_task-Sleep_acq-headband",
                            "subject_id": "sub-1",
                            "task_id": "Sleep",
                            "channel_map_id": "map1",
                            "participant_info": {"age": "25", "sex": "F"},
                        },
                        {
                            "recording_id": "sub-2_task-Sleep_acq-headband",
                            "subject_id": "sub-2",
                            "task_id": "Sleep",
                            "channel_map_id": "map1",
                            "participant_info": {"age": "30", "sex": "M"},
                        },
                    ]
                },
            }
        }

    @pytest.fixture
    def temp_raw_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @patch("brainsets.utils.open_neuro_pipeline.OpenNeuroEEGPipeline._load_config")
    @patch("brainsets.utils.open_neuro_pipeline.validate_dataset_id")
    @patch("brainsets.utils.open_neuro_pipeline.construct_s3_url")
    def test_get_manifest_basic(
        self,
        mock_construct_s3_url,
        mock_validate_dataset_id,
        mock_load_config,
        mock_config,
        temp_raw_dir,
    ):
        mock_validate_dataset_id.return_value = "ds006695"
        mock_load_config.return_value = mock_config
        mock_construct_s3_url.side_effect = (
            lambda dataset_id, recording_id: f"s3://openneuro.org/{dataset_id}/{recording_id}"
        )

        TestPipeline.dataset_id = "ds006695"
        TestPipeline.version_tag = None
        TestPipeline.subject_ids = None

        manifest = TestPipeline.get_manifest(temp_raw_dir, None)

        assert isinstance(manifest, pd.DataFrame)
        assert len(manifest) == 2
        assert "recording_id" not in manifest.columns
        assert manifest.index.name == "recording_id"
        assert "dataset_id" in manifest.columns
        assert "version_tag" in manifest.columns
        assert "subject_id" in manifest.columns
        assert "subject_info" in manifest.columns
        assert "task_id" in manifest.columns
        assert "channel_mapping" in manifest.columns
        assert "device_info" in manifest.columns
        assert "s3_url" in manifest.columns
        assert "fpath" in manifest.columns

        first_row = manifest.iloc[0]
        assert first_row["dataset_id"] == "ds006695"
        assert first_row["version_tag"] == "1.0.0"
        assert first_row["subject_id"] == "sub-1"
        assert first_row["task_id"] == "Sleep"
        assert isinstance(first_row["subject_info"], dict)
        assert first_row["subject_info"]["age"] == "25"

    @patch("brainsets.utils.open_neuro_pipeline.OpenNeuroEEGPipeline._load_config")
    @patch("brainsets.utils.open_neuro_pipeline.validate_dataset_id")
    @patch("brainsets.utils.open_neuro_pipeline.construct_s3_url")
    def test_get_manifest_with_subject_filter(
        self,
        mock_construct_s3_url,
        mock_validate_dataset_id,
        mock_load_config,
        mock_config,
        temp_raw_dir,
    ):
        mock_validate_dataset_id.return_value = "ds006695"
        mock_load_config.return_value = mock_config
        mock_construct_s3_url.side_effect = (
            lambda dataset_id, recording_id: f"s3://openneuro.org/{dataset_id}/{recording_id}"
        )

        TestPipeline.dataset_id = "ds006695"
        TestPipeline.version_tag = None
        TestPipeline.subject_ids = ["sub-1"]

        manifest = TestPipeline.get_manifest(temp_raw_dir, None)

        assert isinstance(manifest, pd.DataFrame)
        assert len(manifest) == 1
        assert manifest.iloc[0]["subject_id"] == "sub-1"

    @patch("brainsets.utils.open_neuro_pipeline.OpenNeuroEEGPipeline._load_config")
    @patch("brainsets.utils.open_neuro_pipeline.validate_dataset_id")
    def test_get_manifest_no_recordings(
        self, mock_validate_dataset_id, mock_load_config, temp_raw_dir
    ):
        mock_validate_dataset_id.return_value = "ds006695"
        mock_load_config.return_value = {
            "dataset": {
                "metadata": {"name": "Test Dataset", "version": "1.0.0"},
                "recording_info": {"recording_info": []},
            }
        }

        TestPipeline.dataset_id = "ds006695"
        TestPipeline.version_tag = None
        TestPipeline.subject_ids = None

        with pytest.raises(ValueError, match="No recording_info found"):
            TestPipeline.get_manifest(temp_raw_dir, None)

    @patch("brainsets.utils.open_neuro_pipeline.OpenNeuroEEGPipeline._load_config")
    @patch("brainsets.utils.open_neuro_pipeline.validate_dataset_id")
    def test_get_manifest_missing_version(
        self, mock_validate_dataset_id, mock_load_config, temp_raw_dir
    ):
        mock_validate_dataset_id.return_value = "ds006695"
        mock_load_config.return_value = {
            "dataset": {
                "metadata": {
                    "name": "Test Dataset",
                },
                "recording_info": {
                    "recording_info": [
                        {
                            "recording_id": "sub-1_task-Sleep_acq-headband",
                            "subject_id": "sub-1",
                            "task_id": "Sleep",
                        }
                    ]
                },
            }
        }

        TestPipeline.dataset_id = "ds006695"
        TestPipeline.version_tag = None
        TestPipeline.subject_ids = None

        with pytest.raises(ValueError, match="Version number not found"):
            TestPipeline.get_manifest(temp_raw_dir, None)


class TestDownload:
    """Test cases for download method."""

    @pytest.fixture
    def pipeline_instance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            processed_dir = Path(tmpdir) / "processed"
            return TestPipeline(raw_dir, processed_dir, None)

    @pytest.fixture
    def mock_manifest_item(self):
        item = pd.Series(
            {
                "dataset_id": "ds006695",
                "version_tag": "1.0.0",
                "subject_id": "sub-1",
                "subject_info": {"age": "25", "sex": "F"},
                "task_id": "Sleep",
                "channel_mapping": {"Fp1": ("Fp1", "EEG")},
                "device_info": {
                    "device_name": "Test Device",
                    "device_manufacturer": "Test Manufacturer",
                },
                "s3_url": "s3://openneuro.org/ds006695/sub-1_task-Sleep_acq-headband",
                "fpath": Path("/tmp/raw/sub-1"),
            },
            name="sub-1_task-Sleep_acq-headband",
        )

        mock_item = MagicMock()
        mock_item.dataset_id = item["dataset_id"]
        mock_item.version_tag = item["version_tag"]
        mock_item.s3_url = item["s3_url"]
        mock_item.Index = item.name
        mock_item.subject_id = item["subject_id"]
        mock_item.subject_info = item["subject_info"]
        mock_item.task_id = item["task_id"]
        mock_item.device_info = item["device_info"]
        mock_item.channel_mapping = item["channel_mapping"]
        mock_item.fpath = item["fpath"]
        return mock_item

    @patch("brainsets.utils.open_neuro_pipeline.check_recording_files_exist")
    @patch("brainsets.utils.open_neuro_pipeline.download_prefix_from_s3")
    def test_download_new_file(
        self,
        mock_download_prefix,
        mock_check_files,
        pipeline_instance,
        mock_manifest_item,
    ):
        mock_check_files.return_value = False
        mock_download_prefix.return_value = None

        result = pipeline_instance.download(mock_manifest_item)

        assert isinstance(result, dict)
        assert result["recording_id"] == "sub-1_task-Sleep_acq-headband"
        assert result["subject_id"] == "sub-1"
        assert result["subject_info"] == {"age": "25", "sex": "F"}
        assert result["device_info"] == {
            "device_name": "Test Device",
            "device_manufacturer": "Test Manufacturer",
        }
        assert result["channel_mapping"] == {"Fp1": ("Fp1", "EEG")}
        assert "fpath" in result

        mock_download_prefix.assert_called_once()
        mock_check_files.assert_called_once()

    @patch("brainsets.utils.open_neuro_pipeline.check_recording_files_exist")
    @patch("brainsets.utils.open_neuro_pipeline.download_prefix_from_s3")
    def test_download_file_exists(
        self,
        mock_download_prefix,
        mock_check_files,
        pipeline_instance,
        mock_manifest_item,
    ):
        mock_check_files.return_value = True

        subject_dir = pipeline_instance.raw_dir / "sub-1"
        subject_dir.mkdir(parents=True, exist_ok=True)

        result = pipeline_instance.download(mock_manifest_item)

        assert isinstance(result, dict)
        assert result["recording_id"] == "sub-1_task-Sleep_acq-headband"
        assert result["subject_id"] == "sub-1"

        mock_download_prefix.assert_not_called()
        mock_check_files.assert_called_once()

    @patch("brainsets.utils.open_neuro_pipeline.check_recording_files_exist")
    @patch("brainsets.utils.open_neuro_pipeline.download_prefix_from_s3")
    def test_download_with_redownload_flag(
        self,
        mock_download_prefix,
        mock_check_files,
        pipeline_instance,
        mock_manifest_item,
    ):
        mock_check_files.return_value = True
        mock_download_prefix.return_value = None

        pipeline_instance.args = Namespace(redownload=True)

        subject_dir = pipeline_instance.raw_dir / "sub-1"
        subject_dir.mkdir(parents=True, exist_ok=True)

        result = pipeline_instance.download(mock_manifest_item)

        mock_download_prefix.assert_called_once()

    @patch("brainsets.utils.open_neuro_pipeline.check_recording_files_exist")
    @patch("brainsets.utils.open_neuro_pipeline.download_prefix_from_s3")
    def test_download_error_handling(
        self,
        mock_download_prefix,
        mock_check_files,
        pipeline_instance,
        mock_manifest_item,
    ):
        mock_check_files.return_value = False
        mock_download_prefix.side_effect = Exception("Download failed")

        with pytest.raises(RuntimeError, match="Failed to download"):
            pipeline_instance.download(mock_manifest_item)
