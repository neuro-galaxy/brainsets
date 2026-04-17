"""Unit tests for OpenNeuro Pipeline classes."""

import pytest
import numpy as np
import pandas as pd
import mne

from pathlib import Path
from unittest.mock import MagicMock, patch, Mock, PropertyMock
from argparse import Namespace

from temporaldata import Data, Interval

from brainsets.utils.openneuro.pipeline import (
    OpenNeuroPipeline,
    OpenNeuroEEGPipeline,
    OpenNeuroIEEGPipeline,
)
from brainsets.descriptions import (
    BrainsetDescription,
    SubjectDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.taxonomy import Species


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files."""
    return tmp_path


@pytest.fixture
def mock_args_no_reprocessing():
    """Mock args with redownload and reprocess set to False."""
    return Namespace(redownload=False, reprocess=False)


@pytest.fixture
def mock_args_with_reprocessing():
    """Mock args with redownload and reprocess set to True."""
    return Namespace(redownload=True, reprocess=True)


@pytest.fixture
def manifest_row():
    """Mock manifest row with typical structure."""
    row = MagicMock()
    row.Index = "rec-001"
    row.subject_id = "sub-01"
    row.s3_url = "s3://openneuro.org/ds005085/sub-01/eeg/rec-001"
    return row


@pytest.fixture
def participants_df():
    """Mock participants DataFrame."""
    return pd.DataFrame(
        {
            "participant_id": ["sub-01", "sub-02"],
            "age": [25, 30],
            "sex": ["M", "F"],
        }
    ).set_index("participant_id")


@pytest.fixture
def mock_raw():
    """Mock MNE raw object."""
    raw = MagicMock(spec=mne.io.BaseRaw)
    raw.info = {"sfreq": 250.0, "meas_date": None, "bads": ["EEG_BAD"]}
    raw.get_data.return_value = np.random.randn(10, 1000)
    raw.ch_names = [
        "EEG_1",
        "EEG_2",
        "EEG_3",
        "EEG_IGNORE",
        "EEG_BAD",
        "EOG_L",
        "EOG_R",
        "EMG",
        "STIM",
    ] + ["unused"] * 3
    raw.get_channel_types.return_value = ["eeg"] * len(raw.ch_names)
    raw.times = np.linspace(0, 4, 1000)
    raw.get_montage.return_value = None
    return raw


@pytest.fixture
def eeg_pipeline_class():
    """Concrete EEG pipeline class for testing."""

    class TestEEGPipeline(OpenNeuroEEGPipeline):
        dataset_id = "ds005085"
        brainset_id = "test_eeg_brainset"
        origin_version = "1.0.0"

    return TestEEGPipeline


@pytest.fixture
def ieeg_pipeline_class():
    """Concrete iEEG pipeline class for testing."""

    class TestIEEGPipeline(OpenNeuroIEEGPipeline):
        dataset_id = "ds005085"
        brainset_id = "test_ieeg_brainset"
        origin_version = "1.0.0"

    return TestIEEGPipeline


@pytest.fixture
def eeg_pipeline_instance(eeg_pipeline_class, temp_dir, mock_args_no_reprocessing):
    """Instantiated EEG pipeline."""
    instance = eeg_pipeline_class(
        raw_dir=temp_dir / "raw",
        processed_dir=temp_dir / "processed",
        args=mock_args_no_reprocessing,
    )
    return instance


@pytest.fixture
def ieeg_pipeline_instance(ieeg_pipeline_class, temp_dir, mock_args_no_reprocessing):
    """Instantiated iEEG pipeline."""
    instance = ieeg_pipeline_class(
        raw_dir=temp_dir / "raw",
        processed_dir=temp_dir / "processed",
        args=mock_args_no_reprocessing,
    )
    return instance


# ============================================================================
# Tests for OpenNeuroEEGPipeline
# ============================================================================


class TestOpenNeuroEEGPipeline:
    """Tests for OpenNeuroEEGPipeline class."""

    def test_modality_is_eeg(self, eeg_pipeline_class):
        """EEG pipeline has 'eeg' modality."""
        assert eeg_pipeline_class.modality == "eeg"

    def test_required_attributes_present(self, eeg_pipeline_class):
        """Pipeline has required class attributes."""
        assert eeg_pipeline_class.dataset_id == "ds005085"
        assert eeg_pipeline_class.brainset_id == "test_eeg_brainset"
        assert eeg_pipeline_class.origin_version == "1.0.0"


class TestOpenNeuroIEEGPipeline:
    """Tests for OpenNeuroIEEGPipeline class."""

    def test_modality_is_ieeg(self, ieeg_pipeline_class):
        """iEEG pipeline has 'ieeg' modality."""
        assert ieeg_pipeline_class.modality == "ieeg"

    def test_required_attributes_present(self, ieeg_pipeline_class):
        """Pipeline has required class attributes."""
        assert ieeg_pipeline_class.dataset_id == "ds005085"
        assert ieeg_pipeline_class.brainset_id == "test_ieeg_brainset"
        assert ieeg_pipeline_class.origin_version == "1.0.0"


# ============================================================================
# Tests for get_manifest
# ============================================================================


class TestGetManifest:
    """Tests for get_manifest class method."""

    @patch("brainsets.utils.openneuro.pipeline.validate_dataset_id")
    @patch("brainsets.utils.openneuro.pipeline.validate_dataset_version")
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_eeg_success(
        self,
        mock_fetch_eeg,
        mock_fetch_files,
        mock_validate_version,
        mock_validate_id,
        eeg_pipeline_class,
        temp_dir,
    ):
        """get_manifest successfully generates manifest for EEG dataset."""
        mock_validate_id.return_value = "ds005085"
        mock_validate_version.return_value = "1.0.0"
        # Make the file list longer (simulate more files returned by fetch_all_filenames)
        mock_fetch_files.return_value = [
            "sub-01/eeg/sub-01_task-rest_eeg.edf",
            "sub-01/eeg/sub-01_task-math_eeg.edf",
            "sub-02/eeg/sub-02_task-rest_eeg.edf",
            "participants.tsv",
            "README",
        ]
        mock_fetch_eeg.return_value = [
            {
                "subject_id": "sub-01",
                "recording_id": "sub-01_task-rest",
                "fpath": "sub-01/eeg/sub-01_task-rest_eeg.edf",
            },
            {
                "subject_id": "sub-01",
                "recording_id": "sub-01_task-math",
                "fpath": "sub-01/eeg/sub-01_task-math_eeg.edf",
            },
            {
                "subject_id": "sub-02",
                "recording_id": "sub-02_task-rest",
                "fpath": "sub-02/eeg/sub-02_task-rest_eeg.edf",
            },
        ]

        with patch(
            "brainsets.utils.openneuro.pipeline.construct_s3_url_from_path"
        ) as mock_s3:
            # We want the s3_url to depend on what is being called with, so use side_effect.
            def s3_url_side_effect(dataset_id, fpath, recording_id):
                # mimic the actual s3_url format
                parent_dir = str(Path(fpath).parent)
                return f"s3://openneuro.org/{dataset_id}/{parent_dir}/{recording_id}"

            mock_s3.side_effect = s3_url_side_effect

            result = eeg_pipeline_class.get_manifest(temp_dir, None)

        assert isinstance(result, pd.DataFrame)
        for rec in [
            ("sub-01_task-rest", "sub-01", "sub-01/eeg", "sub-01_task-rest"),
            ("sub-01_task-math", "sub-01", "sub-01/eeg", "sub-01_task-math"),
            ("sub-02_task-rest", "sub-02", "sub-02/eeg", "sub-02_task-rest"),
        ]:
            recording_id, subject_id, sub_dir, rec_id = rec
            assert recording_id in result.index
            assert result.loc[recording_id, "subject_id"] == subject_id
            expected_s3_url = f"s3://openneuro.org/ds005085/{sub_dir}/{rec_id}"
            assert result.loc[recording_id, "s3_url"] == expected_s3_url

    @patch("brainsets.utils.openneuro.pipeline.validate_dataset_id")
    @patch("brainsets.utils.openneuro.pipeline.validate_dataset_version")
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_ieeg_recordings")
    def test_get_manifest_ieeg_success(
        self,
        mock_fetch_ieeg,
        mock_fetch_files,
        mock_validate_version,
        mock_validate_id,
        ieeg_pipeline_class,
        temp_dir,
    ):
        """get_manifest successfully generates manifest for iEEG dataset."""
        mock_validate_id.return_value = "ds005085"
        mock_validate_version.return_value = "1.0.0"
        # Make the file list longer (simulate more files returned by fetch_all_filenames for ieeg)
        mock_fetch_files.return_value = [
            "sub-01/ieeg/sub-01_task-rest_ieeg.edf",
            "sub-01/ieeg/sub-01_task-math_ieeg.edf",
            "sub-02/ieeg/sub-02_task-rest_ieeg.edf",
            "participants.tsv",
            "README",
        ]
        mock_fetch_ieeg.return_value = [
            {
                "subject_id": "sub-01",
                "recording_id": "sub-01_task-rest",
                "fpath": "sub-01/ieeg/sub-01_task-rest_ieeg.edf",
            },
            {
                "subject_id": "sub-01",
                "recording_id": "sub-01_task-math",
                "fpath": "sub-01/ieeg/sub-01_task-math_ieeg.edf",
            },
            {
                "subject_id": "sub-02",
                "recording_id": "sub-02_task-rest",
                "fpath": "sub-02/ieeg/sub-02_task-rest_ieeg.edf",
            },
        ]

        with patch(
            "brainsets.utils.openneuro.pipeline.construct_s3_url_from_path"
        ) as mock_s3:
            # Make the s3_url depend on what is being called with, just like in the eeg test
            def s3_url_side_effect(dataset_id, fpath, recording_id):
                parent_dir = str(Path(fpath).parent)
                return f"s3://openneuro.org/{dataset_id}/{parent_dir}/{recording_id}"

            mock_s3.side_effect = s3_url_side_effect

            result = ieeg_pipeline_class.get_manifest(temp_dir, None)

        assert isinstance(result, pd.DataFrame)
        for rec in [
            ("sub-01_task-rest", "sub-01", "sub-01/ieeg", "sub-01_task-rest"),
            ("sub-01_task-math", "sub-01", "sub-01/ieeg", "sub-01_task-math"),
            ("sub-02_task-rest", "sub-02", "sub-02/ieeg", "sub-02_task-rest"),
        ]:
            recording_id, subject_id, sub_dir, rec_id = rec
            assert recording_id in result.index
            assert result.loc[recording_id, "subject_id"] == subject_id
            expected_s3_url = f"s3://openneuro.org/ds005085/{sub_dir}/{rec_id}"
            assert result.loc[recording_id, "s3_url"] == expected_s3_url

    @patch("brainsets.utils.openneuro.pipeline.validate_dataset_id")
    @patch("brainsets.utils.openneuro.pipeline.validate_dataset_version")
    @patch("brainsets.utils.openneuro.pipeline.validate_subject_ids")
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_with_subject_filter(
        self,
        mock_fetch_eeg,
        mock_fetch_files,
        mock_validate_subject_ids,
        mock_validate_version,
        mock_validate_id,
        temp_dir,
    ):
        """get_manifest filters recordings by subject_ids when specified."""

        class FilteredPipeline(OpenNeuroEEGPipeline):
            dataset_id = "ds005085"
            brainset_id = "test_eeg"
            origin_version = "1.0.0"
            subject_ids = ["sub-01"]

        mock_validate_id.return_value = "ds005085"
        mock_validate_version.return_value = "1.0.0"
        mock_validate_subject_ids.return_value = ["sub-01"]
        mock_fetch_files.return_value = [
            "sub-01/eeg/rec-001_eeg.edf",
            "sub-02/eeg/rec-001_eeg.edf",
        ]
        mock_fetch_eeg.return_value = [
            {
                "subject_id": "sub-01",
                "recording_id": "rec-001",
                "fpath": "sub-01/eeg/rec-001_eeg.edf",
            },
            {
                "subject_id": "sub-02",
                "recording_id": "rec-001",
                "fpath": "sub-02/eeg/rec-001_eeg.edf",
            },
        ]

        with patch(
            "brainsets.utils.openneuro.pipeline.construct_s3_url_from_path"
        ) as mock_s3:
            mock_s3.return_value = "https://example.com/rec-001"
            result = FilteredPipeline.get_manifest(temp_dir, None)

        assert len(result) == 1
        assert result.index[0] == "rec-001"
        assert result.iloc[0]["subject_id"] == "sub-01"
        assert result.iloc[0]["s3_url"] == "https://example.com/rec-001"

    @patch("brainsets.utils.openneuro.pipeline.validate_dataset_id")
    @patch("brainsets.utils.openneuro.pipeline.validate_dataset_version")
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    def test_get_manifest_raises_on_unknown_modality(
        self, mock_fetch_files, mock_validate_version, mock_validate_id, temp_dir
    ):
        """get_manifest raises ValueError for unknown modality."""

        class BadPipeline(OpenNeuroPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            modality = "unknown"

        mock_validate_id.return_value = "ds005085"
        mock_validate_version.return_value = "1.0.0"
        mock_fetch_files.return_value = []

        with pytest.raises(ValueError, match="Unknown modality"):
            BadPipeline.get_manifest(temp_dir, None)

    @patch("brainsets.utils.openneuro.pipeline.validate_dataset_id")
    @patch("brainsets.utils.openneuro.pipeline.validate_dataset_version")
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_raises_on_no_recordings_found(
        self,
        mock_fetch_eeg,
        mock_fetch_files,
        mock_validate_version,
        mock_validate_id,
        eeg_pipeline_class,
        temp_dir,
    ):
        """get_manifest raises ValueError when no recordings are found."""
        mock_validate_id.return_value = "ds005085"
        mock_validate_version.return_value = "1.0.0"
        mock_fetch_files.return_value = []
        mock_fetch_eeg.return_value = []

        with pytest.raises(ValueError, match="No EEG recordings found"):
            eeg_pipeline_class.get_manifest(temp_dir, None)

    @patch("brainsets.utils.openneuro.pipeline.validate_dataset_id")
    @patch("brainsets.utils.openneuro.pipeline.validate_dataset_version")
    @patch("brainsets.utils.openneuro.pipeline.validate_subject_ids")
    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_raises_on_no_matching_subjects(
        self,
        mock_fetch_eeg,
        mock_fetch_files,
        mock_validate_subject_ids,
        mock_validate_version,
        mock_validate_id,
        temp_dir,
    ):
        """get_manifest raises ValueError when no subjects match the filter."""

        class FilteredPipeline(OpenNeuroEEGPipeline):
            dataset_id = "ds005085"
            brainset_id = "test_eeg"
            origin_version = "1.0.0"
            subject_ids = ["sub-99"]

        mock_validate_id.return_value = "ds005085"
        mock_validate_version.return_value = "1.0.0"
        mock_validate_subject_ids.return_value = ["sub-99"]
        mock_fetch_files.return_value = ["sub-01/eeg/rec-001.edf"]
        mock_fetch_eeg.return_value = [
            {
                "subject_id": "sub-01",
                "recording_id": "rec-001",
                "fpath": "sub-01/eeg/rec-001.edf",
            },
        ]

        with pytest.raises(ValueError, match="No recordings were found"):
            FilteredPipeline.get_manifest(temp_dir, None)


# ============================================================================
# Tests for download
# ============================================================================


class TestDownload:
    """Tests for download method."""

    def test_download_creates_raw_dir(self, eeg_pipeline_instance, manifest_row):
        """download creates raw_dir if it doesn't exist."""
        with patch("brainsets.utils.openneuro.pipeline.download_recording"):
            with patch(
                "brainsets.utils.openneuro.pipeline.download_dataset_description"
            ):
                with patch(
                    "brainsets.utils.openneuro.pipeline.check_eeg_recording_files_exist",
                    return_value=False,
                ):
                    eeg_pipeline_instance.download(manifest_row)

        assert eeg_pipeline_instance.raw_dir.exists()

    def test_download_returns_dict_with_required_keys(
        self, eeg_pipeline_instance, manifest_row
    ):
        """download returns dict with subject_id and recording_id."""
        with patch("brainsets.utils.openneuro.pipeline.download_recording"):
            with patch(
                "brainsets.utils.openneuro.pipeline.download_dataset_description"
            ):
                with patch(
                    "brainsets.utils.openneuro.pipeline.check_eeg_recording_files_exist",
                    return_value=False,
                ):
                    result = eeg_pipeline_instance.download(manifest_row)

        assert "subject_id" in result
        assert "recording_id" in result
        assert result["subject_id"] == "sub-01"
        assert result["recording_id"] == "rec-001"

    def test_download_eeg_skips_if_files_exist_and_no_redownload(
        self, temp_dir, mock_args_no_reprocessing, eeg_pipeline_class, manifest_row
    ):
        """download skips if files exist and redownload is False."""
        pipeline = eeg_pipeline_class(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=mock_args_no_reprocessing,
        )

        with patch(
            "brainsets.utils.openneuro.pipeline.check_eeg_recording_files_exist",
            return_value=True,
        ):
            with patch(
                "brainsets.utils.openneuro.pipeline.download_recording"
            ) as mock_download:
                result = pipeline.download(manifest_row)

        mock_download.assert_not_called()
        assert result["recording_id"] == "rec-001"

    def test_download_ieeg_skips_if_files_exist_and_no_redownload(
        self, temp_dir, mock_args_no_reprocessing, ieeg_pipeline_class, manifest_row
    ):
        """download skips for iEEG if files exist and redownload is False."""
        pipeline = ieeg_pipeline_class(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=mock_args_no_reprocessing,
        )

        with patch(
            "brainsets.utils.openneuro.pipeline.check_ieeg_recording_files_exist",
            return_value=True,
        ):
            with patch(
                "brainsets.utils.openneuro.pipeline.download_recording"
            ) as mock_download:
                result = pipeline.download(manifest_row)

        mock_download.assert_not_called()
        assert result["recording_id"] == "rec-001"

    def test_download_eeg_redownloads_when_redownload_true(
        self, temp_dir, mock_args_with_reprocessing, eeg_pipeline_class, manifest_row
    ):
        """download redownloads when redownload=True even if files exist."""
        pipeline = eeg_pipeline_class(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=mock_args_with_reprocessing,
        )

        with patch(
            "brainsets.utils.openneuro.pipeline.download_recording"
        ) as mock_download:
            with patch(
                "brainsets.utils.openneuro.pipeline.download_dataset_description"
            ):
                result = pipeline.download(manifest_row)

        mock_download.assert_called_once()

    def test_download_raises_on_s3_error(self, eeg_pipeline_instance, manifest_row):
        """download raises RuntimeError on download failure."""
        with patch(
            "brainsets.utils.openneuro.pipeline.download_recording",
            side_effect=Exception("S3 error"),
        ):
            with patch(
                "brainsets.utils.openneuro.pipeline.check_eeg_recording_files_exist",
                return_value=False,
            ):
                with pytest.raises(RuntimeError, match="Failed to download"):
                    eeg_pipeline_instance.download(manifest_row)


# ============================================================================
# Tests for channel remapping methods
# ============================================================================


class TestChannelRemapping:
    """Tests for channel remapping methods."""

    def test_get_channel_name_remapping_returns_class_attribute(
        self, eeg_pipeline_instance
    ):
        """get_channel_name_remapping returns CHANNEL_NAME_REMAPPING when defined."""

        class CustomEEGPipeline(OpenNeuroEEGPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            CHANNEL_NAME_REMAPPING = {"PSG_F3": "F3", "PSG_F4": "F4"}

        pipeline = CustomEEGPipeline(
            raw_dir=Path("/tmp/raw"),
            processed_dir=Path("/tmp/processed"),
            args=Namespace(redownload=False, reprocess=False),
        )
        assert pipeline.get_channel_name_remapping() == {"PSG_F3": "F3", "PSG_F4": "F4"}

    def test_get_channel_name_remapping_returns_none_by_default(
        self, eeg_pipeline_instance
    ):
        """get_channel_name_remapping returns None when not defined."""
        assert eeg_pipeline_instance.get_channel_name_remapping() is None

    def test_get_channel_name_remapping_accepts_recording_id(
        self, eeg_pipeline_instance
    ):
        """get_channel_name_remapping accepts recording_id parameter."""
        result = eeg_pipeline_instance.get_channel_name_remapping(
            recording_id="rec-001"
        )
        assert result is None

    def test_get_type_channels_remapping_returns_class_attribute(self):
        """get_type_channels_remapping returns TYPE_CHANNELS_REMAPPING when defined."""

        class CustomEEGPipeline(OpenNeuroEEGPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            TYPE_CHANNELS_REMAPPING = {"EEG": ["F3", "F4"], "EOG": ["EOG"]}

        pipeline = CustomEEGPipeline(
            raw_dir=Path("/tmp/raw"),
            processed_dir=Path("/tmp/processed"),
            args=Namespace(redownload=False, reprocess=False),
        )
        assert pipeline.get_type_channels_remapping() == {
            "EEG": ["F3", "F4"],
            "EOG": ["EOG"],
        }

    def test_get_type_channels_remapping_returns_none_by_default(
        self, eeg_pipeline_instance
    ):
        """get_type_channels_remapping returns None when not defined."""
        assert eeg_pipeline_instance.get_type_channels_remapping() is None

    def test_get_type_channels_remapping_accepts_recording_id(
        self, eeg_pipeline_instance
    ):
        """get_type_channels_remapping accepts recording_id parameter."""
        result = eeg_pipeline_instance.get_type_channels_remapping(
            recording_id="rec-001"
        )
        assert result is None


# ============================================================================
# Tests for generate_splits
# ============================================================================


class TestGenerateSplits:
    """Tests for generate_splits method."""

    def test_generate_splits_creates_train_and_valid_intervals(
        self, eeg_pipeline_instance
    ):
        """generate_splits creates train and valid intervals."""
        starts = np.array([0.0])
        ends = np.array([100.0])
        domain = Interval(start=starts, end=ends)

        result = eeg_pipeline_instance.generate_splits(
            domain=domain,
            subject_id="sub-01",
            session_id="ses-01",
        )

        assert isinstance(result, Data)
        assert hasattr(result, "train")
        assert hasattr(result, "valid")

    def test_generate_splits_respects_split_ratios(self, temp_dir, eeg_pipeline_class):
        """generate_splits respects custom split_ratios."""

        class CustomRatioPipeline(OpenNeuroEEGPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            split_ratios = (0.8, 0.2)

        pipeline = CustomRatioPipeline(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=Namespace(redownload=False, reprocess=False),
        )

        starts = np.array([0.0])
        ends = np.array([100.0])
        domain = Interval(start=starts, end=ends)

        result = pipeline.generate_splits(domain, "sub-01", "ses-01")

        train_duration = result.train.end[0] - result.train.start[0]
        assert np.isclose(train_duration, 80.0)

    def test_generate_splits_raises_on_invalid_split_ratios_length(
        self, eeg_pipeline_instance
    ):
        """generate_splits raises ValueError if split_ratios length != 2."""
        eeg_pipeline_instance.split_ratios = (0.5, 0.3, 0.2)
        starts = np.array([0.0])
        ends = np.array([100.0])
        domain = Interval(start=starts, end=ends)

        with pytest.raises(ValueError, match="must contain exactly two values"):
            eeg_pipeline_instance.generate_splits(domain, "sub-01", "ses-01")

    def test_generate_splits_raises_on_negative_split_ratios(
        self, eeg_pipeline_instance
    ):
        """generate_splits raises ValueError if any split ratio is negative."""
        eeg_pipeline_instance.split_ratios = (0.5, -0.5)
        starts = np.array([0.0])
        ends = np.array([100.0])
        domain = Interval(start=starts, end=ends)

        with pytest.raises(ValueError, match="cannot contain negative values"):
            eeg_pipeline_instance.generate_splits(domain, "sub-01", "ses-01")

    def test_generate_splits_raises_on_non_unit_sum(self, eeg_pipeline_instance):
        """generate_splits raises ValueError if split ratios don't sum to 1.0."""
        eeg_pipeline_instance.split_ratios = (0.5, 0.3)
        starts = np.array([0.0])
        ends = np.array([100.0])
        domain = Interval(start=starts, end=ends)

        with pytest.raises(ValueError, match="must sum to 1.0"):
            eeg_pipeline_instance.generate_splits(domain, "sub-01", "ses-01")

    def test_generate_splits_creates_assignment_splits(self, eeg_pipeline_instance):
        """generate_splits creates intersubject and intersession assignments."""
        starts = np.array([0.0])
        ends = np.array([100.0])
        domain = Interval(start=starts, end=ends)

        result = eeg_pipeline_instance.generate_splits(domain, "sub-01", "ses-01")

        assert hasattr(result, "intersubject_assignment")
        assert hasattr(result, "intersession_assignment")


# ============================================================================
# Tests for _process_common
# ============================================================================


class TestProcessCommon:
    """Tests for _process_common method."""

    @patch("brainsets.utils.openneuro.pipeline.MNE_BIDS_AVAILABLE", False)
    def test_process_common_raises_when_mne_bids_unavailable(
        self, eeg_pipeline_instance
    ):
        """_process_common raises ImportError if mne-bids not available."""
        download_output = {"recording_id": "rec-001", "subject_id": "sub-01"}

        with pytest.raises(ImportError, match="mne-bids"):
            eeg_pipeline_instance._process_common(download_output)

    @patch("brainsets.utils.openneuro.pipeline.read_raw_bids")
    def test_process_common_skips_if_already_processed(
        self, mock_read_raw, eeg_pipeline_instance
    ):
        """_process_common returns None if file already exists and not reprocessing."""
        eeg_pipeline_instance.processed_dir.mkdir(exist_ok=True, parents=True)
        (eeg_pipeline_instance.processed_dir / "rec-001.h5").touch()
        download_output = {"recording_id": "rec-001", "subject_id": "sub-01"}

        result = eeg_pipeline_instance._process_common(download_output)

        assert result is None
        mock_read_raw.assert_not_called()

    @patch("brainsets.utils.openneuro.pipeline.build_bids_path")
    @patch("brainsets.utils.openneuro.pipeline.read_raw_bids")
    @patch("brainsets.utils.openneuro.pipeline.extract_signal")
    @patch("brainsets.utils.openneuro.pipeline.extract_channels")
    @patch("brainsets.utils.openneuro.pipeline.extract_measurement_date")
    @patch("brainsets.utils.openneuro.pipeline.get_subject_info")
    def test_process_common_creates_data_object(
        self,
        mock_subject_info,
        mock_meas_date,
        mock_extract_channels,
        mock_extract_signal,
        mock_read_raw,
        mock_bids_path,
        eeg_pipeline_instance,
        mock_raw,
    ):
        """_process_common creates and returns Data object."""
        eeg_pipeline_instance.processed_dir.mkdir(exist_ok=True, parents=True)
        download_output = {"recording_id": "rec-001", "subject_id": "sub-01"}

        # Mock return values
        mock_bids_path.return_value = "path/to/bids"
        mock_read_raw.return_value = mock_raw
        mock_subject_info.return_value = {"age": 25, "sex": "M"}
        mock_meas_date.return_value = "2023-01-01"
        mock_extract_signal.return_value = MagicMock(
            domain=Interval(start=np.array([0.0]), end=np.array([100.0]))
        )
        mock_extract_channels.return_value = {}

        with patch.object(
            eeg_pipeline_instance, "generate_splits", return_value=MagicMock()
        ):
            with patch.object(
                eeg_pipeline_instance, "_participants_data", new_callable=PropertyMock
            ) as mock_participants:
                mock_participants.return_value = None
                result = eeg_pipeline_instance._process_common(download_output)

        assert result is not None
        assert len(result) == 2
        data, store_path = result
        assert isinstance(data, Data)
        assert isinstance(store_path, Path)

    @patch("brainsets.utils.openneuro.pipeline.build_bids_path")
    @patch("brainsets.utils.openneuro.pipeline.read_raw_bids")
    @patch("brainsets.utils.openneuro.pipeline.get_subject_info")
    @patch("brainsets.utils.openneuro.pipeline.extract_measurement_date")
    @patch("brainsets.utils.openneuro.pipeline.extract_signal")
    def test_channel_and_type_remapping_and_ignore_channels(
        self,
        mock_extract_signal,
        mock_meas_date,
        mock_subject_info,
        mock_read_raw,
        mock_bids_path,
        temp_dir,
        mock_args_no_reprocessing,
        mock_raw,
    ):
        """Test that _process_common remaps channel names/types and applies ignore channels."""

        # Dummy pipeline with remapping and ignore channel logic
        class CustomEEGPipeline(OpenNeuroEEGPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            CHANNEL_NAME_REMAPPING = {"EEG_1": "F3", "EEG_4": "F4"}
            TYPE_CHANNELS_REMAPPING = {
                "EEG": ["F3", "EEG_2", "EEG_3", "F4"],
                "EOG": ["EOG_L", "EOG_R", "EOG"],
                "EMG": ["EMG"],
                "STIM": ["STIM"],
                "MISC": ["unused"],
            }
            IGNORE_CHANNELS = ["EEG_IGNORE"]

        pipeline = CustomEEGPipeline(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=mock_args_no_reprocessing,
        )

        # Set up test instance and mock methods
        download_output = {"recording_id": "rec-001", "subject_id": "sub-01"}

        # Mock return values
        mock_bids_path.return_value = "path/to/bids_path"
        mock_read_raw.return_value = mock_raw
        mock_subject_info.return_value = {"age": 25, "sex": "M"}
        mock_meas_date.return_value = "2023-01-01"
        mock_extract_signal.return_value = MagicMock(
            domain=Interval(start=np.array([0.0]), end=np.array([100.0]))
        )

        # Run
        with patch.object(pipeline, "generate_splits", return_value=MagicMock()):
            with patch.object(
                pipeline, "_participants_data", new_callable=PropertyMock
            ) as mock_participants:
                mock_participants.return_value = None
                data, path = pipeline._process_common(download_output)

        # Ensure remapping and ignore channels were respected in extract_channels call
        assert all(
            data.channels.id
            == np.array(
                ["F3", "EEG_2", "EEG_3", "EEG_BAD", "EOG_L", "EOG_R", "EMG", "STIM"]
                + ["unused"] * 3
            )
        )
        assert all(
            data.channels.type
            == np.array(
                [
                    "eeg",
                    "eeg",
                    "eeg",
                    "eeg",
                    "eog",
                    "eog",
                    "emg",
                    "stim",
                    "misc",
                    "misc",
                    "misc",
                ]
            )
        )
        assert all(
            data.channels.bad
            == np.array(
                [
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ]
            )
        )


# ============================================================================
# Tests for process
# ============================================================================


class TestProcess:
    """Tests for process method."""

    def test_process_skips_when_process_common_returns_none(
        self, eeg_pipeline_instance
    ):
        """process returns early if _process_common returns None."""
        download_output = {"recording_id": "rec-001", "subject_id": "sub-01"}

        with patch.object(eeg_pipeline_instance, "_process_common", return_value=None):
            with patch("builtins.open") as mock_file:
                eeg_pipeline_instance.process(download_output)

        mock_file.assert_not_called()

    def test_process_saves_data_to_h5(self, eeg_pipeline_instance):
        """process saves Data object to HDF5 file."""
        download_output = {"recording_id": "rec-001", "subject_id": "sub-01"}
        mock_data = MagicMock(spec=Data)
        store_path = eeg_pipeline_instance.processed_dir / "rec-001.h5"

        with patch.object(
            eeg_pipeline_instance,
            "_process_common",
            return_value=(mock_data, store_path),
        ):
            with patch("brainsets.utils.openneuro.pipeline.h5py.File"):
                eeg_pipeline_instance.process(download_output)

        mock_data.to_hdf5.assert_called_once()
