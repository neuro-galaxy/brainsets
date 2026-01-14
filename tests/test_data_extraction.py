"""Tests for brainsets.utils.open_neuro_utils.data_extraction module."""

import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest
from temporaldata import Interval

from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.taxonomy import Sex, Species
from brainsets.utils.open_neuro_utils.data_extraction import (
    channels_from_modality_map,
    extract_brainset_description,
    extract_channels,
    extract_device_description,
    extract_meas_date,
    extract_session_description,
    extract_signal,
    extract_subject_description,
    generate_train_valid_splits_one_epoch,
)


class TestExtractBrainsetDescription:
    def test_creates_description(self):
        result = extract_brainset_description(
            dataset_id="ds005555",
            origin_version="1.0.0",
            derived_version="2.0.0",
            source="https://openneuro.org",
            description="Test dataset",
        )

        assert isinstance(result, BrainsetDescription)
        assert result.id == "ds005555"
        assert result.origin_version == "1.0.0"
        assert result.derived_version == "2.0.0"
        assert result.source == "https://openneuro.org"
        assert result.description == "Test dataset"


class TestExtractSubjectDescription:
    def test_with_all_fields(self):
        result = extract_subject_description(
            subject_id="sub-01",
            age=25,
            sex="M",
        )

        assert isinstance(result, SubjectDescription)
        assert result.id == "sub-01"
        assert result.species == Species.HOMO_SAPIENS
        assert result.age == 25.0
        assert result.sex == Sex.MALE

    def test_with_none_values(self):
        result = extract_subject_description(
            subject_id="sub-01",
            age=None,
            sex=None,
        )

        assert result.id == "sub-01"
        assert result.age == 0.0
        assert result.sex == Sex.UNKNOWN

    def test_with_string_age(self):
        result = extract_subject_description(
            subject_id="sub-01",
            age="30",
            sex="F",
        )

        assert result.age == 30.0
        assert result.sex == Sex.FEMALE

    def test_with_invalid_string_age(self):
        result = extract_subject_description(
            subject_id="sub-01",
            age="invalid",
            sex=None,
        )

        assert result.age == 0.0

    def test_with_sex_enum(self):
        result = extract_subject_description(
            subject_id="sub-01",
            age=25,
            sex=Sex.OTHER,
        )

        assert result.sex == Sex.OTHER

    def test_with_sex_int(self):
        result = extract_subject_description(
            subject_id="sub-01",
            age=25,
            sex=2,
        )

        assert result.sex == Sex.FEMALE


class TestExtractSessionDescription:
    def test_creates_description(self):
        recording_date = datetime.datetime(2024, 1, 15, 10, 30, 0)
        result = extract_session_description(
            session_id="sub-01_ses-01",
            recording_date=recording_date,
        )

        assert isinstance(result, SessionDescription)
        assert result.id == "sub-01_ses-01"
        assert result.recording_date == recording_date


class TestExtractDeviceDescription:
    def test_creates_description(self):
        result = extract_device_description(device_id="device-01")

        assert isinstance(result, DeviceDescription)
        assert result.id == "device-01"


class TestExtractMeasDate:
    def test_with_date(self):
        mock_raw = MagicMock()
        mock_raw.info = {"meas_date": datetime.datetime(2024, 1, 15)}

        result = extract_meas_date(mock_raw)

        assert result == datetime.datetime(2024, 1, 15)

    def test_without_date(self):
        mock_raw = MagicMock()
        mock_raw.info = {"meas_date": None}

        result = extract_meas_date(mock_raw)

        assert result is None


class TestExtractSignal:
    def test_extracts_signal(self):
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.get_data.return_value = np.random.randn(4, 1000)

        result = extract_signal(mock_raw)

        assert result.sampling_rate == 256.0
        assert result.signal.shape == (1000, 4)


class TestExtractChannels:
    def test_extracts_channels(self):
        mock_raw = MagicMock()
        mock_raw.ch_names = ["F3", "F4", "C3", "C4"]
        mock_raw.get_channel_types.return_value = ["eeg", "eeg", "eeg", "eeg"]

        result = extract_channels(mock_raw)

        assert len(result.id) == 4
        assert list(result.id) == ["F3", "F4", "C3", "C4"]
        assert list(result.types) == ["eeg", "eeg", "eeg", "eeg"]


class TestChannelsFromModalityMap:
    def test_creates_channels(self):
        modality_map = {
            "EEG": ["F3", "F4"],
            "EOG": ["EOG_L"],
        }

        result = channels_from_modality_map(modality_map)

        assert len(result.id) == 3
        assert "F3" in result.id
        assert "F4" in result.id
        assert "EOG_L" in result.id

    def test_empty_map(self):
        result = channels_from_modality_map({})

        assert len(result.id) == 0
        assert len(result.types) == 0


class TestGenerateTrainValidSplits:
    def test_default_split(self):
        epoch = Interval(start=np.array([0.0]), end=np.array([100.0]))

        train, valid = generate_train_valid_splits_one_epoch(epoch)

        assert train.start[0] == 0.0
        assert train.end[0] == 90.0
        assert valid.start[0] == 90.0
        assert valid.end[0] == 100.0

    def test_custom_split(self):
        epoch = Interval(start=np.array([0.0]), end=np.array([100.0]))

        train, valid = generate_train_valid_splits_one_epoch(epoch, [0.8, 0.2])

        assert train.start[0] == 0.0
        assert train.end[0] == 80.0
        assert valid.start[0] == 80.0
        assert valid.end[0] == 100.0

    def test_invalid_epoch_length(self):
        epoch = Interval(
            start=np.array([0.0, 50.0]),
            end=np.array([50.0, 100.0]),
        )

        with pytest.raises(ValueError, match="single interval"):
            generate_train_valid_splits_one_epoch(epoch)

    def test_invalid_split_ratios(self):
        epoch = Interval(start=np.array([0.0]), end=np.array([100.0]))

        with pytest.raises(ValueError, match="sum to 1"):
            generate_train_valid_splits_one_epoch(epoch, [0.5, 0.3])
