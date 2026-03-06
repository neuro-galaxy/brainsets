import datetime
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

try:
    import mne

    MNE_AVAILABLE = True
    from brainsets.utils.mne_utils import (
        extract_measurement_date,
        extract_eeg_signal,
        extract_channels,
        extract_psg_signal,
        concatenate_recordings,
    )
    from temporaldata import ArrayDict
except ImportError:
    print("MNE not installed")
    MNE_AVAILABLE = False
    extract_measurement_date = None
    extract_eeg_signal = None
    extract_channels = None
    extract_psg_signal = None
    concatenate_recordings = None


def create_mock_raw(
    n_channels=3,
    n_samples=1000,
    sfreq=256.0,
    meas_date=None,
    ch_names=None,
    ch_types=None,
):
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
class TestExtractMeasDate:
    def test_returns_meas_date_when_present(self):
        expected_date = datetime.datetime(
            2023, 6, 15, 10, 30, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw = create_mock_raw(meas_date=expected_date)
        result = extract_measurement_date(mock_raw)
        assert result == expected_date

    def test_returns_unix_epoch_when_meas_date_none(self):
        mock_raw = create_mock_raw(meas_date=None)
        with pytest.warns(UserWarning, match="No measurement date found"):
            result = extract_measurement_date(mock_raw)
        expected = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        assert result == expected


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestExtractEEGSignal:
    def test_returns_regular_time_series(self):
        mock_raw = create_mock_raw(n_channels=4, n_samples=500, sfreq=256.0)
        result = extract_eeg_signal(mock_raw)
        assert hasattr(result, "signal")
        assert hasattr(result, "sampling_rate")
        assert hasattr(result, "domain")

    def test_signal_shape(self):
        n_channels = 4
        n_samples = 500
        mock_raw = create_mock_raw(n_channels=n_channels, n_samples=n_samples)
        result = extract_eeg_signal(mock_raw)
        assert result.signal.shape == (n_samples, n_channels)

    def test_sampling_rate_is_correct(self):
        sfreq = 512.0
        mock_raw = create_mock_raw(sfreq=sfreq)
        result = extract_eeg_signal(mock_raw)
        assert result.sampling_rate == sfreq

    def test_domain_start_is_zero(self):
        mock_raw = create_mock_raw()
        result = extract_eeg_signal(mock_raw)
        assert result.domain.start[0] == 0.0

    def test_domain_end(self):
        n_samples = 1000
        sfreq = 256.0
        mock_raw = create_mock_raw(n_samples=n_samples, sfreq=sfreq)
        result = extract_eeg_signal(mock_raw)
        expected_end = (n_samples - 1) / sfreq
        assert np.isclose(result.domain.end[0], expected_end)

    def test_raises_error_on_empty_signal(self):
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 256.0}
        mock_raw.get_data.return_value = np.array([]).reshape(3, 0)
        with pytest.raises(ValueError, match="Recording contains no samples"):
            extract_eeg_signal(mock_raw)


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestExtractChannels:
    def test_returns_array_dict(self):
        mock_raw = create_mock_raw()

        result = extract_channels(mock_raw)

        assert isinstance(result, ArrayDict)
        assert hasattr(result, "id")
        assert hasattr(result, "type")

    def test_channel_names_extracted_correctly(self):
        expected_names = ["EEG Fpz-Cz", "EOG horizontal", "EMG submental"]
        mock_raw = create_mock_raw(
            ch_names=expected_names, n_channels=len(expected_names)
        )

        result = extract_channels(mock_raw)

        np.testing.assert_array_equal(result.id, np.array(expected_names, dtype="U"))

    def test_channel_types_extracted_correctly(self):
        expected_types = ["eeg", "eog", "emg"]
        mock_raw = create_mock_raw(
            ch_types=expected_types, n_channels=len(expected_types)
        )

        result = extract_channels(mock_raw)

        np.testing.assert_array_equal(result.type, np.array(expected_types, dtype="U"))

    def test_id_dtype_is_unicode(self):
        mock_raw = create_mock_raw()

        result = extract_channels(mock_raw)

        assert result.id.dtype.kind == "U"

    def test_types_dtype_is_unicode(self):
        mock_raw = create_mock_raw()

        result = extract_channels(mock_raw)

        assert result.type.dtype.kind == "U"

    def test_coordinates_not_included_when_montage_missing(self):
        mock_raw = create_mock_raw()

        result = extract_channels(mock_raw)

        assert "x" not in result.keys()
        assert "y" not in result.keys()
        assert "z" not in result.keys()

    def test_coordinates_included_when_montage_has_positions(self):
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

        assert "x" in result.keys()
        assert "y" in result.keys()
        assert "z" in result.keys()
        np.testing.assert_allclose(
            result.x, np.array([0.1, np.nan, 0.4]), equal_nan=True
        )
        np.testing.assert_allclose(
            result.y, np.array([0.2, np.nan, 0.5]), equal_nan=True
        )
        np.testing.assert_allclose(
            result.z, np.array([0.3, np.nan, 0.6]), equal_nan=True
        )


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestExtractPSGSignal:
    def create_mock_psg_raw(self, ch_names, n_samples=1000, sfreq=256.0):
        mock_raw = MagicMock()
        n_channels = len(ch_names)

        mock_raw.info = {"sfreq": sfreq}
        mock_raw.ch_names = ch_names

        mock_data = np.random.randn(n_channels, n_samples)
        mock_times = np.arange(n_samples) / sfreq
        mock_raw.get_data.return_value = (mock_data, mock_times)

        return mock_raw

    def test_extracts_eeg_channels(self):
        ch_names = ["EEG Fpz-Cz", "EEG Pz-Oz", "Other"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        _, channels = extract_psg_signal(mock_raw)

        assert "EEG" in channels.type
        assert np.sum(channels.type == "EEG") == 2

    def test_extracts_eog_channels(self):
        ch_names = ["EOG horizontal", "EEG Fpz-Cz"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        _, channels = extract_psg_signal(mock_raw)

        assert "EOG" in channels.type

    def test_extracts_emg_channels(self):
        ch_names = ["EMG submental", "EEG Fpz-Cz"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        _, channels = extract_psg_signal(mock_raw)

        assert "EMG" in channels.type

    def test_extracts_resp_channels(self):
        ch_names = ["Resp oro-nasal", "EEG Fpz-Cz"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        _, channels = extract_psg_signal(mock_raw)

        assert "RESP" in channels.type

    def test_extracts_temp_channels(self):
        ch_names = ["Temp rectal", "EEG Fpz-Cz"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        _, channels = extract_psg_signal(mock_raw)

        assert "TEMP" in channels.type

    def test_skips_unknown_channels(self):
        ch_names = ["Unknown1", "Unknown2", "EEG Fpz-Cz"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        _, channels = extract_psg_signal(mock_raw)

        assert len(channels.id) == 1  # only EEG

    def test_returns_regular_time_series(self):
        ch_names = ["EEG Fpz-Cz", "EOG horizontal"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        signals, channels = extract_psg_signal(mock_raw)

        assert hasattr(signals, "signal")
        assert hasattr(signals, "sampling_rate")
        assert hasattr(signals, "domain")

    def test_signal_shape(self):
        ch_names = ["EEG Fpz-Cz", "EOG horizontal", "EMG submental"]
        n_samples = 500
        mock_raw = self.create_mock_psg_raw(ch_names, n_samples=n_samples)

        signals, _ = extract_psg_signal(mock_raw)

        assert signals.signal.shape == (n_samples, 3)

    def test_sampling_rate_correct(self):
        ch_names = ["EEG Fpz-Cz", "EOG horizontal"]
        sfreq = 128.0
        mock_raw = self.create_mock_psg_raw(ch_names, sfreq=sfreq)

        signals, _ = extract_psg_signal(mock_raw)

        assert signals.sampling_rate == sfreq

    def test_channels_has_id_and_type(self):
        ch_names = ["EEG Fpz-Cz", "EOG horizontal"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        _, channels = extract_psg_signal(mock_raw)

        assert hasattr(channels, "id")
        assert hasattr(channels, "type")

    def test_raises_error_when_no_signals_extracted(self):
        ch_names = ["Unknown1", "Unknown2"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        with pytest.raises(ValueError, match="No signals extracted from PSG file"):
            extract_psg_signal(mock_raw)

    def test_channel_ids_preserved(self):
        ch_names = ["EEG Fpz-Cz", "EOG horizontal"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        _, channels = extract_psg_signal(mock_raw)

        assert "EEG Fpz-Cz" in channels.id
        assert "EOG horizontal" in channels.id

    def test_extracts_fpz_cz_pattern_case_insensitive(self):
        ch_names = ["FPZ-CZ", "fpz-cz", "Fpz-Cz"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        _, channels = extract_psg_signal(mock_raw)

        assert np.sum(channels.type == "EEG") == 3

    def test_extracts_pz_oz_pattern_case_insensitive(self):
        ch_names = ["PZ-OZ", "pz-oz", "Pz-Oz"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        _, channels = extract_psg_signal(mock_raw)

        assert np.sum(channels.type == "EEG") == 3

    def test_domain_uses_first_and_last_time_values(self):
        ch_names = ["EEG Fpz-Cz", "EOG horizontal"]
        n_samples = 1000
        sfreq = 100.0
        mock_raw = self.create_mock_psg_raw(ch_names, n_samples=n_samples, sfreq=sfreq)
        _, times = mock_raw.get_data(return_times=True)

        signals, _ = extract_psg_signal(mock_raw)

        assert np.isclose(signals.domain.start, times[0])
        assert np.isclose(signals.domain.end, times[-1])


@pytest.mark.skipif(not MNE_AVAILABLE, reason="mne not installed")
class TestConcatenateRecordings:
    def test_empty_list_raises_error(self):
        with pytest.raises(ValueError, match="recordings list cannot be empty"):
            concatenate_recordings([])

    def test_invalid_policy_raises_error(self):
        mock_raw = create_mock_raw()
        with pytest.raises(ValueError, match="on_mismatch must be one of"):
            concatenate_recordings([mock_raw], on_mismatch="invalid")

    def test_non_list_input_raises_error(self):
        mock_raw = create_mock_raw()
        with pytest.raises(ValueError, match="recordings must be a list"):
            concatenate_recordings(mock_raw)

    def test_non_raw_object_raises_error(self):
        with pytest.raises(ValueError, match="is not an MNE Raw-like object"):
            concatenate_recordings([{"not": "raw"}])

    def test_single_recording_concatenates(self):
        meas_date = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        mock_raw = create_mock_raw(n_channels=3, n_samples=1000, meas_date=meas_date)
        mock_raw.copy.return_value = mock_raw

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = mock_raw
            result = concatenate_recordings([mock_raw])
            mock_concat.assert_called_once()
            assert result == mock_raw

    def test_multiple_recordings_same_meas_date_concatenates(self):
        meas_date = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        mock_raw1 = create_mock_raw(n_channels=3, n_samples=1000, meas_date=meas_date)
        mock_raw2 = create_mock_raw(n_channels=3, n_samples=1000, meas_date=meas_date)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_result = MagicMock()
            mock_concat.return_value = mock_result
            result = concatenate_recordings([mock_raw1, mock_raw2])
            mock_concat.assert_called_once()
            # Verify copies were passed to concatenate_raws
            call_args = mock_concat.call_args[0][0]
            assert len(call_args) == 2
            assert result == mock_result

    def test_recordings_sorted_by_meas_date(self):
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
            # Pass them in non-chronological order: date1, date2, date3
            result = concatenate_recordings([mock_raw1, mock_raw2, mock_raw3])

            # Verify they were sorted before concatenation: date3, date1, date2
            call_args = mock_concat.call_args[0][0]
            assert call_args[0] == mock_raw3
            assert call_args[1] == mock_raw1
            assert call_args[2] == mock_raw2

    def test_different_meas_dates_raise_with_raise_policy(self):
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 11, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)

        with pytest.raises(ValueError, match="Measurement dates are not uniform"):
            concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="raise")

    def test_different_meas_dates_warn_with_warn_policy(self):
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 11, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            with pytest.warns(UserWarning, match="Measurement dates are not uniform"):
                concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="warn")

    def test_different_meas_dates_ignore_with_ignore_policy(self):
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 11, 0, 0, tzinfo=datetime.timezone.utc)

        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="ignore")
            mock_concat.assert_called_once()

    def test_different_channel_names_raise_with_raise_policy(self):
        meas_date = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        mock_raw1 = create_mock_raw(ch_names=["CH0", "CH1", "CH2"], meas_date=meas_date)
        mock_raw2 = create_mock_raw(ch_names=["CH0", "CH1", "CH3"], meas_date=meas_date)

        with pytest.raises(ValueError, match="Channel names/order differ"):
            concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="raise")

    def test_different_channel_names_warn_with_warn_policy(self):
        meas_date = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        mock_raw1 = create_mock_raw(ch_names=["CH0", "CH1", "CH2"], meas_date=meas_date)
        mock_raw2 = create_mock_raw(ch_names=["CH0", "CH1", "CH3"], meas_date=meas_date)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            with pytest.warns(UserWarning, match="Channel names/order differ"):
                concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="warn")

    def test_different_channel_names_ignore_with_ignore_policy(self):
        meas_date = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        mock_raw1 = create_mock_raw(ch_names=["CH0", "CH1", "CH2"], meas_date=meas_date)
        mock_raw2 = create_mock_raw(ch_names=["CH0", "CH1", "CH3"], meas_date=meas_date)
        mock_raw1.copy.return_value = mock_raw1
        mock_raw2.copy.return_value = mock_raw2

        with patch("brainsets.utils.mne_utils.mne.concatenate_raws") as mock_concat:
            mock_concat.return_value = MagicMock()
            concatenate_recordings([mock_raw1, mock_raw2], on_mismatch="ignore")
            mock_concat.assert_called_once()

    def test_default_policy_is_raise(self):
        date1 = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
        date2 = datetime.datetime(2023, 6, 15, 11, 0, 0, tzinfo=datetime.timezone.utc)
        mock_raw1 = create_mock_raw(meas_date=date1)
        mock_raw2 = create_mock_raw(meas_date=date2)

        with pytest.raises(ValueError):
            concatenate_recordings([mock_raw1, mock_raw2])

    def test_concatenate_raws_called_with_copies(self):
        meas_date = datetime.datetime(2023, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
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


class TestCheckMneAvailable:
    """Test the _check_mne_available function when MNE is not available."""

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_extract_measurement_date_raises_mne_import_error(self):
        from brainsets.utils.mne_utils import extract_measurement_date

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="extract_measurement_date requires the MNE library"
        ):
            extract_measurement_date(mock_raw)

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_extract_eeg_signal_raises_mne_import_error(self):
        from brainsets.utils.mne_utils import extract_eeg_signal

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="extract_eeg_signal requires the MNE library"
        ):
            extract_eeg_signal(mock_raw)

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_extract_channels_raises_mne_import_error(self):
        from brainsets.utils.mne_utils import extract_channels

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="extract_channels requires the MNE library"
        ):
            extract_channels(mock_raw)

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_extract_psg_signal_raises_mne_import_error(self):
        from brainsets.utils.mne_utils import extract_psg_signal

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="extract_psg_signal requires the MNE library"
        ):
            extract_psg_signal(mock_raw)

    @patch("brainsets.utils.mne_utils.MNE_AVAILABLE", False)
    def test_concatenate_recordings_raises_mne_import_error(self):
        from brainsets.utils.mne_utils import concatenate_recordings

        mock_raw = MagicMock()
        with pytest.raises(
            ImportError, match="concatenate_recordings requires the MNE library"
        ):
            concatenate_recordings([mock_raw])
