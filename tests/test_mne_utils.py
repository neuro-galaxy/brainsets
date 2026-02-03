import datetime
import numpy as np
import pytest
from unittest.mock import MagicMock

pytest.importorskip("mne")
from brainsets.utils.mne_utils import (
    extract_meas_date,
    extract_eeg_signal,
    extract_channels,
    extract_psg_signal,
)


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
    }
    mock_raw.ch_names = ch_names
    mock_raw.get_channel_types.return_value = ch_types

    mock_data = np.random.randn(n_channels, n_samples)
    mock_raw.get_data.return_value = mock_data

    return mock_raw


class TestExtractMeasDate:
    def test_returns_meas_date_when_present(self):
        expected_date = datetime.datetime(
            2023, 6, 15, 10, 30, 0, tzinfo=datetime.timezone.utc
        )
        mock_raw = create_mock_raw(meas_date=expected_date)
        result = extract_meas_date(mock_raw)
        assert result == expected_date

    def test_returns_unix_epoch_when_meas_date_none(self):
        mock_raw = create_mock_raw(meas_date=None)
        with pytest.warns(UserWarning, match="No measurement date found"):
            result = extract_meas_date(mock_raw)
        expected = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        assert result == expected


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


class TestExtractChannels:

    def test_returns_array_dict(self):
        mock_raw = create_mock_raw()
        result = extract_channels(mock_raw)
        assert hasattr(result, "ids")
        assert hasattr(result, "types")

    def test_channel_names_extracted_correctly(self):
        ch_names = ["Fp1", "Fp2", "C3", "C4"]
        mock_raw = create_mock_raw(ch_names=ch_names, n_channels=len(ch_names))

        result = extract_channels(mock_raw)

        np.testing.assert_array_equal(result.ids, np.array(ch_names, dtype="U"))

    def test_channel_types_extracted_correctly(self):
        ch_types = ["eeg", "eeg", "eog", "emg"]
        mock_raw = create_mock_raw(ch_types=ch_types, n_channels=len(ch_types))

        result = extract_channels(mock_raw)

        np.testing.assert_array_equal(result.types, np.array(ch_types, dtype="U"))

    def test_id_dtype_is_unicode(self):
        mock_raw = create_mock_raw()

        result = extract_channels(mock_raw)

        assert result.ids.dtype.kind == "U"

    def test_types_dtype_is_unicode(self):
        mock_raw = create_mock_raw()

        result = extract_channels(mock_raw)

        assert result.types.dtype.kind == "U"


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

        signals, units = extract_psg_signal(mock_raw)

        assert "EEG" in units.modality
        assert np.sum(units.modality == "EEG") == 2

    def test_extracts_eog_channels(self):
        ch_names = ["EOG horizontal", "EEG Fpz-Cz"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        signals, units = extract_psg_signal(mock_raw)

        assert "EOG" in units.modality

    def test_extracts_emg_channels(self):
        ch_names = ["EMG submental", "EEG Fpz-Cz"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        signals, units = extract_psg_signal(mock_raw)

        assert "EMG" in units.modality

    def test_extracts_resp_channels(self):
        ch_names = ["Resp oro-nasal", "EEG Fpz-Cz"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        signals, units = extract_psg_signal(mock_raw)

        assert "RESP" in units.modality

    def test_extracts_temp_channels(self):
        ch_names = ["Temp rectal", "EEG Fpz-Cz"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        signals, units = extract_psg_signal(mock_raw)

        assert "TEMP" in units.modality

    def test_skips_unknown_channels(self):
        ch_names = ["Unknown1", "Unknown2", "EEG Fpz-Cz"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        signals, units = extract_psg_signal(mock_raw)

        assert len(units.id) == 1  # only EEG

    def test_returns_regular_time_series(self):
        ch_names = ["EEG Fpz-Cz", "EOG horizontal"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        signals, units = extract_psg_signal(mock_raw)

        assert hasattr(signals, "signal")
        assert hasattr(signals, "sampling_rate")
        assert hasattr(signals, "domain")

    def test_signal_shape(self):
        ch_names = ["EEG Fpz-Cz", "EOG horizontal", "EMG submental"]
        n_samples = 500
        mock_raw = self.create_mock_psg_raw(ch_names, n_samples=n_samples)

        signals, units = extract_psg_signal(mock_raw)

        assert signals.signal.shape == (n_samples, 3)

    def test_sampling_rate_correct(self):
        ch_names = ["EEG Fpz-Cz"]
        sfreq = 128.0
        mock_raw = self.create_mock_psg_raw(ch_names, sfreq=sfreq)

        signals, units = extract_psg_signal(mock_raw)

        assert signals.sampling_rate == sfreq

    def test_units_has_id_and_modality(self):
        ch_names = ["EEG Fpz-Cz", "EOG horizontal"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        signals, units = extract_psg_signal(mock_raw)

        assert hasattr(units, "id")
        assert hasattr(units, "modality")

    def test_raises_error_when_no_signals_extracted(self):
        ch_names = ["Unknown1", "Unknown2"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        with pytest.raises(ValueError, match="No signals extracted from PSG file"):
            extract_psg_signal(mock_raw)

    def test_channel_ids_preserved(self):
        ch_names = ["EEG Fpz-Cz", "EOG horizontal"]
        mock_raw = self.create_mock_psg_raw(ch_names)

        signals, units = extract_psg_signal(mock_raw)

        assert "EEG Fpz-Cz" in units.id
        assert "EOG horizontal" in units.id
