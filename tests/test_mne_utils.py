import datetime
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

try:
    import mne

    MNE_AVAILABLE = True
    from brainsets.utils.mne_utils import (
        extract_measurement_date,
        extract_signal,
        extract_channels,
        extract_psg_signal,
        concatenate_recordings,
    )
    from temporaldata import ArrayDict
except ImportError:
    print("TEMPORALDATA_NOT_AVAILABLE")
    MNE_AVAILABLE = False
    extract_measurement_date = None
    extract_signal = None
    extract_channels = None
    extract_psg_signal = None
    concatenate_recordings = None
    ArrayDict = None


def create_mock_raw(
    n_channels=3,
    n_samples=1000,
    sfreq=256.0,
    meas_date=None,
    ch_names=None,
    ch_types=None,
):
    """Helper to create a mock MNE Raw object for testing."""
    mock_raw = MagicMock()

    if ch_names is None:
        ch_names = [f"CH{i}" for i in range(n_channels)]
    if ch_types is None:
        ch_types = ["eeg"] * n_channels

    mock_raw.info = {
        "sfreq": sfreq,
        "meas_date": meas_date,
        "bads": [],
    }
    mock_raw.ch_names = ch_names
    mock_raw.get_channel_types.return_value = ch_types
    mock_raw.get_montage.return_value = None

    mock_data = np.random.randn(n_channels, n_samples)
    mock_raw.get_data.return_value = mock_data

    return mock_raw


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestExtractMeasurementDate:
    """Test extraction of measurement date from MNE Raw objects."""

    def test_returns_meas_date_when_present(self):
        """Test that the measurement date is returned when present."""
        expected_date = datetime.datetime(
            2023, 6, 15, 10, 30, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw = create_mock_raw(meas_date=expected_date)
        result = extract_measurement_date(mock_raw)
        assert result == expected_date

    def test_returns_unix_epoch_when_meas_date_none(self):
        """Test that Unix epoch is returned when measurement date is missing."""
        mock_raw = create_mock_raw(meas_date=None)
        with pytest.warns(UserWarning, match="No measurement date found"):
            result = extract_measurement_date(mock_raw)
        expected = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        assert result == expected

    def test_preserves_timezone_info(self):
        """Test that timezone information is preserved."""
        tz = datetime.timezone(datetime.timedelta(hours=5))
        expected_date = datetime.datetime(2023, 6, 15, 10, 30, 0, tzinfo=tz)
        mock_raw = create_mock_raw(meas_date=expected_date)
        result = extract_measurement_date(mock_raw)
        assert result.tzinfo == tz


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestExtractSignal:
    """Test extraction of time-series signal from MNE Raw objects."""

    def test_returns_regular_time_series(self):
        """Test that a RegularTimeSeries object is returned."""
        mock_raw = create_mock_raw(n_channels=4, n_samples=500, sfreq=256.0)
        result = extract_signal(mock_raw)
        assert hasattr(result, "signal")
        assert hasattr(result, "sampling_rate")
        assert hasattr(result, "domain")

    def test_signal_shape_is_samples_by_channels(self):
        """Test that signal shape is (n_samples, n_channels)."""
        n_channels = 4
        n_samples = 500
        mock_raw = create_mock_raw(n_channels=n_channels, n_samples=n_samples)
        result = extract_signal(mock_raw)
        assert result.signal.shape == (n_samples, n_channels)

    def test_sampling_rate_matches_mne_sfreq(self):
        """Test that sampling rate is correctly extracted."""
        sfreq = 512.0
        mock_raw = create_mock_raw(sfreq=sfreq)
        result = extract_signal(mock_raw)
        assert result.sampling_rate == sfreq

    def test_domain_start_is_zero(self):
        """Test that domain start is 0."""
        mock_raw = create_mock_raw()
        result = extract_signal(mock_raw)
        assert result.domain.start[0] == 0.0

    def test_domain_end_calculation(self):
        """Test that domain end is calculated as (n_samples - 1) / sfreq."""
        n_samples = 1000
        sfreq = 256.0
        mock_raw = create_mock_raw(n_samples=n_samples, sfreq=sfreq)
        result = extract_signal(mock_raw)
        expected_end = (n_samples - 1) / sfreq
        assert np.isclose(result.domain.end[0], expected_end)

    def test_raises_error_when_no_samples(self):
        """Test that ValueError is raised when recording contains no samples."""
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.get_data.return_value = np.array([]).reshape(3, 0)
        with pytest.raises(ValueError, match="Recording contains no samples"):
            extract_signal(mock_raw)

    def test_works_with_single_channel(self):
        """Test extraction with a single channel."""
        mock_raw = create_mock_raw(n_channels=1, n_samples=1000)
        result = extract_signal(mock_raw)
        assert result.signal.shape == (1000, 1)

    def test_works_with_many_channels(self):
        """Test extraction with many channels."""
        n_channels = 64
        n_samples = 500
        mock_raw = create_mock_raw(n_channels=n_channels, n_samples=n_samples)
        result = extract_signal(mock_raw)
        assert result.signal.shape == (n_samples, n_channels)


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestExtractChannels:
    """Test extraction of channel metadata from MNE Raw objects."""

    def test_returns_array_dict(self):
        """Test that an ArrayDict is returned."""
        mock_raw = create_mock_raw()
        result = extract_channels(mock_raw)
        assert isinstance(result, ArrayDict)

    def test_contains_id_and_type_fields(self):
        """Test that returned ArrayDict has 'id' and 'type' fields."""
        mock_raw = create_mock_raw()
        result = extract_channels(mock_raw)
        assert hasattr(result, "id")
        assert hasattr(result, "type")

    def test_id_field_contains_channel_names(self):
        """Test that 'id' field contains the channel names."""
        expected_names = ["EEG Fpz-Cz", "EOG horizontal", "EMG submental"]
        mock_raw = create_mock_raw(
            ch_names=expected_names, n_channels=len(expected_names)
        )
        result = extract_channels(mock_raw)
        np.testing.assert_array_equal(result.id, np.array(expected_names, dtype="U"))

    def test_type_field_contains_channel_types(self):
        """Test that 'type' field contains the channel types."""
        expected_types = ["eeg", "eog", "emg"]
        mock_raw = create_mock_raw(
            ch_types=expected_types, n_channels=len(expected_types)
        )
        result = extract_channels(mock_raw)
        np.testing.assert_array_equal(result.type, np.array(expected_types, dtype="U"))

    def test_id_dtype_is_unicode(self):
        """Test that 'id' field has unicode dtype."""
        mock_raw = create_mock_raw()
        result = extract_channels(mock_raw)
        assert result.id.dtype.kind == "U"

    def test_type_dtype_is_unicode(self):
        """Test that 'type' field has unicode dtype."""
        mock_raw = create_mock_raw()
        result = extract_channels(mock_raw)
        assert result.type.dtype.kind == "U"

    def test_bad_field_omitted_when_no_bad_channels(self):
        """Test that 'bad' field is omitted when there are no bad channels."""
        mock_raw = create_mock_raw()
        result = extract_channels(mock_raw)
        assert not hasattr(result, "bad")

    def test_bad_field_included_when_bad_channels_exist(self):
        """Test that 'bad' field is included and marks bad channels correctly."""
        ch_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))
        mock_raw.info["bads"] = ["CH1"]
        result = extract_channels(mock_raw)
        assert hasattr(result, "bad")
        assert result.bad.dtype == bool
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.bad, expected)

    def test_coord_not_included_when_montage_missing(self):
        """Test that 'coord' field is absent when no montage is available."""
        mock_raw = create_mock_raw()
        result = extract_channels(mock_raw)
        assert not hasattr(result, "coord")

    def test_coord_included_when_montage_has_positions(self):
        """Test that 'coord' field is included when montage positions are available."""
        ch_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))
        montage = MagicMock()
        montage.get_positions.return_value = {
            "ch_pos": {
                "CH0": np.array([0.1, 0.2, 0.3]),
                "CH2": np.array([0.4, 0.5, 0.6]),
            }
        }
        mock_raw.get_montage.return_value = montage

        result = extract_channels(mock_raw)

        assert hasattr(result, "coord")
        assert result.coord.shape == (3, 3)
        np.testing.assert_allclose(
            result.coord,
            np.array([[0.1, 0.2, 0.3], [np.nan, np.nan, np.nan], [0.4, 0.5, 0.6]]),
            equal_nan=True,
        )

    def test_name_mapping_applied(self):
        """Test that channel name mapping is correctly applied."""
        original_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(
            ch_names=original_names, n_channels=len(original_names)
        )
        name_mapping = {"CH0": "NewCH0", "CH1": "NewCH1"}

        result = extract_channels(mock_raw, channels_name_mapping=name_mapping)

        expected = np.array(["NewCH0", "NewCH1", "CH2"], dtype="U")
        np.testing.assert_array_equal(result.id, expected)

    def test_type_mapping_with_original_names(self):
        """Test type mapping using original channel names."""
        ch_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(
            ch_names=ch_names,
            ch_types=["eeg", "eeg", "eeg"],
            n_channels=len(ch_names),
        )
        type_mapping = {"eog": ["CH0"], "emg": ["CH2"]}

        result = extract_channels(mock_raw, channels_type_mapping=type_mapping)

        expected = np.array(["eog", "eeg", "emg"], dtype="U")
        np.testing.assert_array_equal(result.type, expected)

    def test_type_mapping_with_renamed_channels(self):
        """Test type mapping when applied after name mapping."""
        original_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(
            ch_names=original_names,
            ch_types=["eeg", "eeg", "eeg"],
            n_channels=len(original_names),
        )
        name_mapping = {"CH0": "EOG_L", "CH1": "EOG_R", "CH2": "EMG"}
        type_mapping = {"eog": ["EOG_L", "EOG_R"], "emg": ["EMG"]}

        result = extract_channels(
            mock_raw,
            channels_name_mapping=name_mapping,
            channels_type_mapping=type_mapping,
        )

        expected_types = np.array(["eog", "eog", "emg"], dtype="U")
        np.testing.assert_array_equal(result.type, expected_types)

    def test_bad_channels_checked_against_renamed_ids(self):
        """Test that bad channel marking is applied to renamed channel names."""
        original_names = ["CH0", "CH1", "CH2"]
        mock_raw = create_mock_raw(
            ch_names=original_names, n_channels=len(original_names)
        )
        mock_raw.info["bads"] = ["NewCH1"]
        name_mapping = {"CH0": "NewCH0", "CH1": "NewCH1"}

        result = extract_channels(mock_raw, channels_name_mapping=name_mapping)

        assert hasattr(result, "bad")
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.bad, expected)

    def test_montage_extraction_failure_graceful_fallback(self):
        """Test that coord extraction failures are logged but don't fail the function."""
        ch_names = ["CH0", "CH1"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))
        montage = MagicMock()
        montage.get_positions.side_effect = RuntimeError("Montage error")
        mock_raw.get_montage.return_value = montage

        result = extract_channels(mock_raw)

        assert hasattr(result, "id")
        assert hasattr(result, "type")
        assert not hasattr(result, "coord")


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestExtractPsgSignal:
    """Test extraction of PSG (polysomnography) signals from MNE Raw objects."""

    @pytest.fixture
    def mock_psg_raw(self):
        """Fixture to create a mock PSG Raw object with EEG, EOG, EMG channels."""
        ch_names = ["EEG Fpz-Cz", "EOG horizontal", "EMG submental", "RESP"]
        n_channels = len(ch_names)
        n_samples = 1000
        sfreq = 256.0

        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": sfreq}
        mock_raw.ch_names = ch_names

        mock_data = np.random.randn(n_channels, n_samples)
        mock_times = np.arange(n_samples) / sfreq
        mock_raw.get_data.return_value = (mock_data, mock_times)

        return mock_raw

    def test_returns_tuple_of_signals_and_channels(self, mock_psg_raw):
        """Test that a tuple of (RegularTimeSeries, ArrayDict) is returned."""
        signals, channels = extract_psg_signal(mock_psg_raw)
        assert hasattr(signals, "signal")
        assert hasattr(signals, "sampling_rate")
        assert hasattr(signals, "domain")
        assert isinstance(channels, ArrayDict)

    def test_extracts_eeg_channels(self):
        """Test that EEG channels are correctly classified."""
        ch_names = ["EEG Fpz-Cz", "EEG Pz-Oz", "Other"]
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.ch_names = ch_names
        n_samples = 100
        mock_data = np.random.randn(len(ch_names), n_samples)
        mock_times = np.arange(n_samples) / 256.0
        mock_raw.get_data.return_value = (mock_data, mock_times)

        _, channels = extract_psg_signal(mock_raw)

        assert "EEG" in channels.type
        assert np.sum(channels.type == "EEG") == 2

    def test_extracts_eog_channels(self):
        """Test that EOG channels are correctly classified."""
        ch_names = ["EOG horizontal", "EEG Fpz-Cz"]
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.ch_names = ch_names
        n_samples = 100
        mock_data = np.random.randn(len(ch_names), n_samples)
        mock_times = np.arange(n_samples) / 256.0
        mock_raw.get_data.return_value = (mock_data, mock_times)

        _, channels = extract_psg_signal(mock_raw)

        assert "EOG" in channels.type

    def test_extracts_emg_channels(self):
        """Test that EMG channels are correctly classified."""
        ch_names = ["EMG submental", "EEG Fpz-Cz"]
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.ch_names = ch_names
        n_samples = 100
        mock_data = np.random.randn(len(ch_names), n_samples)
        mock_times = np.arange(n_samples) / 256.0
        mock_raw.get_data.return_value = (mock_data, mock_times)

        _, channels = extract_psg_signal(mock_raw)

        assert "EMG" in channels.type

    def test_extracts_resp_channels(self):
        """Test that RESP channels are correctly classified."""
        ch_names = ["Resp oro-nasal", "EEG Fpz-Cz"]
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.ch_names = ch_names
        n_samples = 100
        mock_data = np.random.randn(len(ch_names), n_samples)
        mock_times = np.arange(n_samples) / 256.0
        mock_raw.get_data.return_value = (mock_data, mock_times)

        _, channels = extract_psg_signal(mock_raw)

        assert "RESP" in channels.type

    def test_extracts_temp_channels(self):
        """Test that TEMP channels are correctly classified."""
        ch_names = ["Temp rectal", "EEG Fpz-Cz"]
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.ch_names = ch_names
        n_samples = 100
        mock_data = np.random.randn(len(ch_names), n_samples)
        mock_times = np.arange(n_samples) / 256.0
        mock_raw.get_data.return_value = (mock_data, mock_times)

        _, channels = extract_psg_signal(mock_raw)

        assert "TEMP" in channels.type

    def test_skips_unknown_channels(self):
        """Test that channels not matching PSG patterns are skipped."""
        ch_names = ["Unknown1", "Unknown2", "EEG Fpz-Cz"]
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.ch_names = ch_names
        n_samples = 100
        mock_data = np.random.randn(len(ch_names), n_samples)
        mock_times = np.arange(n_samples) / 256.0
        mock_raw.get_data.return_value = (mock_data, mock_times)

        _, channels = extract_psg_signal(mock_raw)

        assert len(channels.id) == 1

    def test_channel_ids_preserved(self, mock_psg_raw):
        """Test that original channel names are preserved in the output."""
        _, channels = extract_psg_signal(mock_psg_raw)
        assert "EEG Fpz-Cz" in channels.id
        assert "EOG horizontal" in channels.id

    def test_fpz_cz_pattern_case_insensitive(self):
        """Test that Fpz-Cz pattern matching is case-insensitive."""
        ch_names = ["FPZ-CZ", "fpz-cz", "Fpz-Cz"]
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.ch_names = ch_names
        n_samples = 100
        mock_data = np.random.randn(len(ch_names), n_samples)
        mock_times = np.arange(n_samples) / 256.0
        mock_raw.get_data.return_value = (mock_data, mock_times)

        _, channels = extract_psg_signal(mock_raw)

        assert np.sum(channels.type == "EEG") == 3

    def test_pz_oz_pattern_case_insensitive(self):
        """Test that Pz-Oz pattern matching is case-insensitive."""
        ch_names = ["PZ-OZ", "pz-oz", "Pz-Oz"]
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.ch_names = ch_names
        n_samples = 100
        mock_data = np.random.randn(len(ch_names), n_samples)
        mock_times = np.arange(n_samples) / 256.0
        mock_raw.get_data.return_value = (mock_data, mock_times)

        _, channels = extract_psg_signal(mock_raw)

        assert np.sum(channels.type == "EEG") == 3

    def test_signal_shape_matches_n_samples_by_n_channels(self, mock_psg_raw):
        """Test that signal shape is (n_samples, n_extracted_channels)."""
        signals, _ = extract_psg_signal(mock_psg_raw)
        n_samples = 1000
        n_extracted = 4  # EEG, EOG, EMG, RESP
        assert signals.signal.shape == (n_samples, n_extracted)

    def test_sampling_rate_extracted_correctly(self, mock_psg_raw):
        """Test that sampling rate is correctly extracted."""
        signals, _ = extract_psg_signal(mock_psg_raw)
        assert signals.sampling_rate == 256.0

    def test_domain_uses_times_from_get_data(self):
        """Test that domain is set using actual times from get_data(return_times=True)."""
        ch_names = ["EEG Fpz-Cz"]
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 100.0}
        mock_raw.ch_names = ch_names
        n_samples = 1000
        mock_data = np.random.randn(1, n_samples)
        mock_times = np.arange(n_samples) / 100.0
        mock_raw.get_data.return_value = (mock_data, mock_times)

        signals, _ = extract_psg_signal(mock_raw)

        assert np.isclose(signals.domain.start, mock_times[0])
        assert np.isclose(signals.domain.end, mock_times[-1])

    def test_raises_error_when_no_signals_extracted(self):
        """Test that ValueError is raised when no matching PSG signals are found."""
        ch_names = ["Unknown1", "Unknown2"]
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.ch_names = ch_names
        n_samples = 100
        mock_data = np.random.randn(len(ch_names), n_samples)
        mock_times = np.arange(n_samples) / 256.0
        mock_raw.get_data.return_value = (mock_data, mock_times)

        with pytest.raises(ValueError, match="No signals extracted from PSG file"):
            extract_psg_signal(mock_raw)

    def test_channels_has_id_and_type(self, mock_psg_raw):
        """Test that returned channels ArrayDict has 'id' and 'type' fields."""
        _, channels = extract_psg_signal(mock_psg_raw)
        assert hasattr(channels, "id")
        assert hasattr(channels, "type")


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestConcatenateRecordings:
    """Test concatenation of multiple MNE Raw objects."""

    def test_empty_list_raises_error(self):
        """Test that ValueError is raised for empty recordings list."""
        with pytest.raises(ValueError, match="Recordings list cannot be empty"):
            concatenate_recordings([])

    def test_non_list_input_raises_error(self):
        """Test that ValueError is raised when input is not a list."""
        mock_raw = create_mock_raw()
        with pytest.raises(ValueError, match="Recordings must be a list"):
            concatenate_recordings(mock_raw)

    def test_non_raw_object_raises_error(self):
        """Test that ValueError is raised for non-Raw-like objects."""
        with pytest.raises(ValueError, match="is not an MNE Raw-like object"):
            concatenate_recordings([{"not": "raw"}])

    def test_invalid_policy_raises_error(self):
        """Test that ValueError is raised for invalid on_mismatch policy."""
        mock_raw = create_mock_raw()
        with pytest.raises(ValueError, match="on_mismatch must be one of"):
            concatenate_recordings([mock_raw], on_mismatch="invalid")

    def test_single_recording_concatenates(self):
        """Test that a single recording can be concatenated."""
        meas_date = datetime.datetime(
            2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw = create_mock_raw(n_channels=3, n_samples=1000, meas_date=meas_date)
        mock_raw.copy.return_value = mock_raw

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = mock_raw
            result = concatenate_recordings([mock_raw])
            mock_concat.assert_called_once()
            assert result == mock_raw

    def test_multiple_recordings_same_meas_date_concatenates(self):
        """Test that multiple recordings with same measurement date are concatenated."""
        meas_date = datetime.datetime(
            2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw1 = create_mock_raw(n_channels=3, n_samples=1000, meas_date=meas_date)
        mock_raw2 = create_mock_raw(n_channels=3, n_samples=1000, meas_date=meas_date)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_result = MagicMock()
            mock_concat.return_value = mock_result
            result = concatenate_recordings([mock_raw1, mock_raw2])
            mock_concat.assert_called_once()
            call_args = mock_concat.call_args[0][0]
            assert len(call_args) == 2
            assert result == mock_result

    def test_recordings_sorted_by_meas_date(self):
        """Test that recordings are sorted by measurement date before concatenation."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 11, 0, 0, tzinfo=datetime.timezone.utc)
        date3 = datetime.datetime(2023, 6, 15, 9, 30, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(n_channels=3, n_samples=500, meas_date=date1)
        mock_raw2 = create_mock_raw(n_channels=3, n_samples=500, meas_date=date2)
        mock_raw3 = create_mock_raw(n_channels=3, n_samples=500, meas_date=date3)

        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2
        mock_raw3.copy.return_value = mock_raw3

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_result = MagicMock()
            mock_concat.return_value = mock_result
            concatenate_recordings([mock_raw1, mock_raw2, mock_raw3])

            call_args = mock_concat.call_args[0][0]
            assert call_args[0] == mock_raw3
            assert call_args[1] == mock_raw1
            assert call_args[2] == mock_raw2

    def test_different_meas_date_days_raise_with_raise_policy(self):
        """Test that ValueError is raised for different measurement date days with raise policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 16, 10, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)

        with pytest.raises(ValueError, match="Measurement days are not uniform"):
            concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="raise")

    def test_different_meas_date_days_warn_with_warn_policy(self):
        """Test that warning is issued for different measurement date days with warn policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 16, 10, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            with pytest.warns(UserWarning, match="Measurement days are not uniform"):
                concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="warn")

    def test_different_meas_date_days_ignore_with_ignore_policy(self):
        """Test that different measurement dates are silently ignored with ignore policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 16, 10, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="ignore")
            mock_concat.assert_called_once()

    def test_different_channel_names_raise_with_raise_policy(self):
        """Test that ValueError is raised for different channel names with raise policy."""
        meas_date = datetime.datetime(
            2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw1 = create_mock_raw(ch_names=["CH0", "CH1", "CH2"], meas_date=meas_date)
        mock_raw2 = create_mock_raw(ch_names=["CH0", "CH1", "CH3"], meas_date=meas_date)

        with pytest.raises(
            ValueError, match="Mismatch in channel names and/or order across recordings"
        ):
            concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="raise")

    def test_different_channel_names_warn_with_warn_policy(self):
        """Test that warning is issued for different channel names with warn policy."""
        meas_date = datetime.datetime(
            2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw1 = create_mock_raw(ch_names=["CH0", "CH1", "CH2"], meas_date=meas_date)
        mock_raw2 = create_mock_raw(ch_names=["CH0", "CH1", "CH3"], meas_date=meas_date)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            with pytest.warns(
                UserWarning, match="Mismatch in channel names and/or order"
            ):
                concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="warn")

    def test_different_channel_names_ignore_with_ignore_policy(self):
        """Test that different channel names are silently ignored with ignore policy."""
        meas_date = datetime.datetime(
            2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw1 = create_mock_raw(ch_names=["CH0", "CH1", "CH2"], meas_date=meas_date)
        mock_raw2 = create_mock_raw(ch_names=["CH0", "CH1", "CH3"], meas_date=meas_date)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="ignore")
            mock_concat.assert_called_once()

    def test_default_policy_is_raise(self):
        """Test that default on_mismatch policy is 'raise'."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 11, 0, 0, tzinfo=datetime.timezone.utc)
        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)

        with pytest.raises(ValueError):
            concatenate_recordings([mock_raw1, mock_raw2])

    def test_invalid_offset_policy_raises_error(self):
        """Test that ValueError is raised for invalid on_offset policy."""
        mock_raw = create_mock_raw()
        with pytest.raises(ValueError, match="on_offset must be one of"):
            concatenate_recordings([mock_raw], on_offset="invalid")

    def test_offset_within_1_hour_succeeds_with_warn_policy(self):
        """Test that recordings within 1 hour offset pass with warn policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 10, 30, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            result = concatenate_recordings([mock_raw1, mock_raw2], on_offset="warn")
            mock_concat.assert_called_once()

    def test_offset_greater_than_1_hour_raise_with_raise_policy(self):
        """Test that ValueError is raised when offset > 1 hour with raise policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)

        with pytest.raises(
            ValueError, match="Offset between recordings .* is greater than 1 hour"
        ):
            concatenate_recordings([mock_raw1, mock_raw2], on_offset="raise")

    def test_offset_greater_than_1_hour_warn_with_warn_policy(self):
        """Test that warning is issued when offset > 1 hour with warn policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            with pytest.warns(
                UserWarning, match="Offset between recordings .* is greater than 1 hour"
            ):
                concatenate_recordings([mock_raw1, mock_raw2], on_offset="warn")

    def test_offset_greater_than_1_hour_ignore_with_ignore_policy(self):
        """Test that offset > 1 hour is silently ignored with ignore policy."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            concatenate_recordings([mock_raw1, mock_raw2], on_offset="ignore")
            mock_concat.assert_called_once()

    def test_default_offset_policy_is_warn(self):
        """Test that default on_offset policy is 'warn'."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            with pytest.warns(
                UserWarning, match="Offset between recordings .* is greater than 1 hour"
            ):
                concatenate_recordings([mock_raw1, mock_raw2])

    def test_multiple_offsets_check_all_consecutive_pairs(self):
        """Test that offset check is applied to all consecutive recording pairs."""
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 11, 30, 0, tzinfo=datetime.timezone.utc)
        date3 = datetime.datetime(2023, 6, 15, 13, 30, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw3 = create_mock_raw(meas_date=date3)

        with pytest.raises(
            ValueError, match="Offset between recordings .* is greater than 1 hour"
        ):
            concatenate_recordings([mock_raw1, mock_raw2, mock_raw3], on_offset="raise")

    def test_concatenate_raws_called_with_copies(self):
        """Test that mne.concatenate_raws is called with copies of recordings."""
        meas_date = datetime.datetime(
            2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw1 = create_mock_raw(meas_date=meas_date)
        mock_raw2 = create_mock_raw(meas_date=meas_date)

        mock_copy1 = MagicMock()
        mock_copy2 = MagicMock()
        mock_raw1.copy.return_value = mock_copy1
        mock_raw2.copy.return_value = mock_copy2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            concatenate_recordings([mock_raw1, mock_raw2])

            call_args = mock_concat.call_args[0][0]
            assert mock_copy1 in call_args
            assert mock_copy2 in call_args


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestCheckMneAvailable:
    """Test that functions raise ImportError when MNE is not available."""

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_extract_measurement_date_raises_import_error(self):
        """Test that extract_measurement_date raises ImportError when MNE is unavailable."""
        from brainsets.utils.mne_utils import extract_measurement_date

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="extract_measurement_date requires the MNE library"
        ):
            extract_measurement_date(mock_raw)

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_extract_signal_raises_import_error(self):
        """Test that extract_signal raises ImportError when MNE is unavailable."""
        from brainsets.utils.mne_utils import extract_signal

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="extract_signal requires the MNE library"
        ):
            extract_signal(mock_raw)

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_extract_channels_raises_import_error(self):
        """Test that extract_channels raises ImportError when MNE is unavailable."""
        from brainsets.utils.mne_utils import extract_channels

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="extract_channels requires the MNE library"
        ):
            extract_channels(mock_raw)

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_extract_psg_signal_raises_import_error(self):
        """Test that extract_psg_signal raises ImportError when MNE is unavailable."""
        from brainsets.utils.mne_utils import extract_psg_signal

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="extract_psg_signal requires the MNE library"
        ):
            extract_psg_signal(mock_raw)

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_concatenate_recordings_raises_import_error(self):
        """Test that concatenate_recordings raises ImportError when MNE is unavailable."""
        from brainsets.utils.mne_utils import concatenate_recordings

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="concatenate_recordings requires the MNE library"
        ):
            concatenate_recordings([mock_raw])
