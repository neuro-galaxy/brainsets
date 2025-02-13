import logging
from tqdm import tqdm

import mne
import pyprep

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from .metrics import (
    hilbert_distance,
    rmse,
)
from .montages import make_montage
from .raw import BrainsetsRaw

from typing import (
    Union,
    List,
    Optional,
    Dict,
)

# Make ABC abstract class
from abc import ABC, abstractmethod

logging.basicConfig(format="%(message)s\n", level=logging.INFO)  # Only show the message


class QCPipeline(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()


class BaseQCPipeline(QCPipeline):
    """
    Base class for Quality Control (QC) Pipelines.
    This class provides a framework for setting up and running quality control
    pipelines on EEG data. It includes methods for setting up channels, montages,
    filters, and Independent Component Analysis (ICA) for artifact removal.
    """

    def __init__(
        self,
        # data
        channels: List[str],
        channels_rename: Optional[Dict[str, str]] = None,
        montage_name: str = "standard_1005",
        # pipeline
        window_length: Union[int, float, None] = 60.0,
        hp_freq: Optional[float] = 1.0,
        lp_freq: Optional[float] = 100.0,
        line_freqs: List[float] = [60.0],
        iclabel_threshold: float = 0.7,
        quality_check: bool = True,
        fit_ica: bool = True,
        quality_check_ica: bool = True,
        include_components: List[str] = ["brain", "other"],
        plot_scores: bool = True,
        random_state: Union[None, int, np.random.RandomState] = None,
    ):

        mne.set_log_level("ERROR")

        # Pipeline setup
        self.channels_rename = channels_rename
        self.window_length = window_length  # seconds
        self.ransac = True
        self.quality_check = quality_check
        self.quality_check_ica = quality_check_ica
        self.plot_scores = plot_scores

        self._setup_channels(channels)
        self._setup_montage(montage_name)
        self._setup_filters(hp_freq, lp_freq, line_freqs)
        self._setup_ica(iclabel_threshold, include_components, fit_ica)

        # Quality check thresholds
        self.oha_threshold = 40e-6
        self.thv_threshold = 40e-6
        self.chv_threshold = 80e-6

        # Set random state
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _to_standard_channel_names(
        self,
        ch_names: List[str],
    ) -> None:
        """Convert channel names to standard channel names."""
        rename_map = {"T3": "T7", "T4": "T8", "P7": "T5", "P8": "T6"}
        ch_names = [rename_map[ch] if ch in rename_map else ch for ch in ch_names]
        return ch_names

    def _setup_channels(
        self,
        ch_names: List[str],
    ) -> None:
        """Setup channel names."""
        self.ch_names = self._to_standard_channel_names(ch_names)

    def _setup_montage(
        self,
        montage_name: str,
    ) -> None:
        """Setup montage."""
        self.montage_name = montage_name
        self.montage = make_montage(
            montage_name=montage_name,
            ch_names=self.ch_names,
        )

    def _setup_filters(
        self,
        hp_freq: Optional[float],
        lp_freq: Optional[float],
        line_freqs: List[float],
    ) -> None:
        """Setup filters."""
        self.hp_freq = hp_freq
        self.lp_freq = lp_freq
        self.line_freqs = np.sort([line_freqs]).flatten()

    def _setup_ica(
        self, iclabel_threshold: float, include_components: List[str], fit_ica: bool
    ) -> None:
        """Setup Independent Component Analysis (ICA) to identify and exclude artifacts."""
        self.include_components = include_components

        # Chech all iclabels are valid
        valid_iclabels = [
            "brain",
            "muscle",
            "eye",
            "heart",
            "line noise",
            "channel noise",
            "other",
        ]
        if not all([label in valid_iclabels for label in self.include_components]):
            raise ValueError("Invalid component label found in include_components.")

        self.iclabel_threshold = iclabel_threshold
        self.fit_ica = fit_ica
        self.ica = None

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def speed_quality_asssessment(
        self,
        raw: Union[mne.io.Raw, BrainsetsRaw],
        verbose: bool = True,
    ) -> bool:
        """Performs a quick quality assessment on raw EEG data.

        This method evaluates the quality of EEG recordings using SPEED criteria:
        - Detection of noisy channels using PREP pipeline criteria
        - Overall High Amplitude (OHA): proportion of samples exceeding amplitude threshold
        - Time-series High Variance (THV): proportion of channels with high temporal variance
        - Channel High Variance (CHV): proportion of timepoints with high spatial variance
        - Bad Channel Ratio (BCR): proportion of channels marked as bad

        Parameters
        ----------
        raw : Union[mne.io.Raw, BrainsetsRaw]
            The raw EEG data to be assessed.
        verbose : bool, default=True
            If True, prints detailed information about the quality assessment results.

        Returns
        -------
        bool
            True if the recording passes all quality criteria:
            - OHA < 0.8 (less than 80% of samples exceed amplitude threshold)
            - THV < 0.5 (less than 50% of channels have high temporal variance)
            - CHV < 0.5 (less than 50% of timepoints have high spatial variance)
            - BCR < 0.8 (less than 80% of channels marked as bad)
            False otherwise.

        Notes
        -----
        The method uses the PREP pipeline's NoisyChannels detection along with
        additional custom metrics to provide a comprehensive quality assessment.
        """
        raw = raw.copy()

        # bad channel detection
        noisychannels = pyprep.NoisyChannels(
            raw,
            do_detrend=False,  # remove low-frequency (<1.0 Hz) trends
            random_state=self.random_state,
        )

        noisychannels.find_bad_by_nan_flat()
        noisychannels.find_bad_by_deviation()
        noisychannels.find_bad_by_hfnoise()
        noisychannels.find_bad_by_correlation(frac_bad=0.01)
        noisychannels.find_bad_by_SNR()

        # calculate quality scores
        oha = np.mean(np.abs(raw._data) > self.oha_threshold)
        thv = np.mean(np.std(raw._data, axis=0) > self.thv_threshold)
        chv = np.mean(np.std(raw._data, axis=1) > self.chv_threshold)

        bad_channels = noisychannels.get_bads()
        bcr = len(bad_channels) / raw.info["nchan"]

        if verbose:
            logging.info(
                f"QC assessment: {len(bad_channels)} out of {raw.info['nchan']} bad channels detected."
            )
            _bad_channels = noisychannels.get_bads(as_dict=True)
            for key, value in _bad_channels.items():
                if value and key != "bad_all":
                    logging.info(f"\t{key}: {value}")

            logging.info(f"\tOHA score = {oha}")
            logging.info(f"\tTHV score = {thv}")
            logging.info(f"\tCHV score = {chv}")
            logging.info(f"\tBCR score = {bcr}")

        return (oha < 0.8) & (thv < 0.5) & (chv < 0.5) & (bcr < 0.8)

    def compute_quality_scores(
        self,
        raw: Union[mne.io.Raw, BrainsetsRaw],
        raw_qc: Union[mne.io.Raw, BrainsetsRaw],
        score: str = "rmse",
        smooth: bool = True,
        norm: bool = True,
    ) -> np.ndarray:
        """
        Compute quality scores between raw and quality-controlled EEG data.

        Parameters
        ----------
        raw : Union[mne.io.Raw, BrainsetsRaw]
            The original raw EEG data.
        raw_qc : Union[mne.io.Raw, BrainsetsRaw]
            The quality-controlled EEG data.
        score : str, optional
            The type of score to compute. Default is "rmse". Options include:
            - "rmse"
            - "hilbert_distance".
        smooth : bool, optional
            Whether to apply smoothing to the computed scores. Default is True.
        norm : bool, optional
            Whether to normalize the computed scores. Default is True.

        Returns
        -------
        np.ndarray
            The computed quality scores.
        """
        raw_signal = raw.copy().get_data(picks=raw_qc.ch_names)
        reconstructed_signal = raw_qc.get_data()

        if score == "hilbert_distance":
            scores = hilbert_distance(
                raw_signal=raw_signal,
                reconstructed_signal=reconstructed_signal,
                smooth=smooth,
                norm=norm,
                sfreq=raw.info["sfreq"],
            )
        elif score == "rmse":
            scores = rmse(
                raw_signal=raw_signal,
                reconstructed_signal=reconstructed_signal,
                smooth=smooth,
                norm=norm,
            )
        else:
            raise ValueError(
                f"Unsupported score metric: {score}. Choose 'rmse' or 'hilbert_distance'."
            )

        return scores

    def get_full_scores(
        self,
        raw: Union[mne.io.Raw, BrainsetsRaw],
        raw_qc: Union[mne.io.Raw, BrainsetsRaw],
        scores: np.ndarray,
    ) -> np.ndarray:
        """Reconstructs full-size scores array by mapping partial scores back to original data dimensions.

        This function takes partial scores computed on a subset of channels (after bad channel removal)
        and maps them back to match the dimensions of the original raw data, filling removed channels
        with zeros.

        Parameters
        ----------
        raw : Union[mne.io.Raw, BrainsetsRaw]
            Original raw data object containing all channels.
        raw_qc : Union[mne.io.Raw, BrainsetsRaw]
            Quality-controlled raw data object with bad channels removed.
        scores : np.ndarray
            Scores array computed on the quality-controlled data, with shape (n_good_channels, n_times).

        Returns
        -------
        np.ndarray
            Full scores array matching original raw data dimensions, with zeros in bad channel positions.
            Shape is (n_total_channels, n_times).

        Raises
        ------
        ValueError
            If number of time points in scores does not match raw data.

        Notes
        -----
        This function is useful for reconstructing full-dimension arrays after computing metrics
        on a subset of channels, maintaining alignment with the original data structure.
        """
        if not scores.shape[1] == raw.get_data().shape[1]:
            raise ValueError(
                (
                    f"Mismatch in the number of time points: 'scores' and 'raw' data must have the same number of time points."
                    f"\n'scores' has {scores.shape[1]} and 'raw' has {raw.get_data().shape[1]} time points."
                )
            )

        # Get original indexes in raw for remaining channels in raw_qc
        # after dropping bad channels
        ch_idx = np.array(
            [
                np.where(np.array(raw.ch_names) == ch)[0][0]
                for ch in np.array(raw_qc.ch_names)
            ]
        )
        full_scores = np.zeros_like(raw.get_data())
        full_scores[np.ix_(ch_idx, np.arange(raw.n_times))] = scores

        return full_scores

    def _plot_scores_and_stacked_signals_segmented(
        self,
        raw_signal: np.ndarray,
        reconstructed_signal: np.ndarray,
        quality_metric: np.ndarray,
        channel_names=None,
        sampling_rate=None,
        time_offset=None,
    ) -> None:
        """
        Plots the stacked time series of multiple channels with quality-based segment coloring.

        Args:
            raw_signal (ndarray): Raw signal of shape (T, N) where T is time steps and N is channels.
            reconstructed_signal (ndarray): Filtered signal of shape (T, N).
            quality_metric (ndarray): Quality metric of shape (T, N) ranging from 0 (bad) to 1 (good).
            channel_names (list): Optional list of channel names. If None, will use Ch 1, Ch 2, etc.
            sampling_rate (float): Optional sampling rate in Hz. If provided, x-axis will show seconds.
        """
        T, N = raw_signal.shape

        if sampling_rate is not None:
            time = np.arange(T) / sampling_rate
            xlabel = "Time (s)"
        else:
            time = np.arange(T)
            xlabel = "Time"

        if time_offset is not None:
            if sampling_rate is not None:
                time = time + (time_offset / sampling_rate)
            else:
                time = time + time_offset

        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

        # Normalize quality metric to colormap
        cmap = plt.get_cmap("RdYlGn")  # Red (bad) to Green (good)
        norm = mcolors.Normalize(vmin=0, vmax=1)

        offset = np.arange(N) * 2.0  # Offset for stacking channels

        for i in range(N):
            # Create a colormap background as an image
            quality_colors = cmap(norm(quality_metric[:, i]))[:, :3]  # Convert to RGB
            quality_image = np.tile(
                quality_colors, (10, 1, 1)
            )  # Expand for visualization

            ax.imshow(
                quality_image,
                extent=[time[0], time[-1], offset[i] - 1, offset[i] + 1],
                aspect="auto",
                alpha=0.5,  # Add transparency
            )

            # Plot raw signal in black
            ax.plot(time, raw_signal[:, i] + offset[i], "k", lw=1)

            # Plot filtered signal as dashed black line
            ax.plot(time, reconstructed_signal[:, i] + offset[i], "k--", lw=1)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Signal Quality")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Channels")
        ax.set_yticks(offset)

        if channel_names is None:
            channel_names = [f"Ch {i+1}" for i in range(N)]
        ax.set_yticklabels(channel_names)

        ax.invert_yaxis()  # Invert to have Ch1 at the top
        plt.show()
        plt.close()

    def plot_quality_scores(
        self,
        raw: Union[mne.io.Raw, BrainsetsRaw],
        raw_baseline: Union[mne.io.Raw, BrainsetsRaw],
        raw_qc: Union[mne.io.Raw, BrainsetsRaw],
        scores: np.ndarray,
    ) -> None:
        np.random.seed(self.random_state)

        # Copy data
        raw_signal = raw.copy()
        raw_baseline = raw_baseline.copy()
        raw_qc = raw_qc.copy()

        # Select 10 random channels from reconstructed signal
        picks_idx = np.random.choice(len(raw_qc.ch_names), 10, replace=False)
        picks_ch_names = np.array(raw_qc.ch_names)[picks_idx]

        # Get idx of selected channels for raw and raw_baseline
        picks_raw_idx = np.array(
            [
                np.where(np.array(raw_signal.ch_names) == ch)[0][0]
                for ch in np.array(picks_ch_names)
            ]
        )
        # picks_raw_baseline_idx = np.array([np.where(np.array(raw_baseline.ch_names) == ch)[0][0] for ch in np.array(picks_ch_names)])

        # Select 1 random 10-sec time slice
        sliced_times = np.array_split(
            np.arange(len(raw_qc.times)),
            int(len(raw_qc.times) / (raw_qc.info["sfreq"] * 5)),
        )
        picks_time = np.random.choice(len(sliced_times), 1, replace=False)

        for i in picks_time:
            start = sliced_times[i][0]
            end = sliced_times[i][-1]

            _raw_baseline = (
                10000 * raw_baseline.get_data(picks=picks_ch_names).T[start:end]
            )
            _raw_qc = 10000 * raw_qc.get_data(picks=picks_ch_names).T[start:end]
            _scores = scores[picks_raw_idx].T[start:end]

            self._plot_scores_and_stacked_signals_segmented(
                raw_signal=_raw_baseline,
                reconstructed_signal=_raw_qc,
                quality_metric=_scores,
                channel_names=picks_ch_names,
                sampling_rate=raw_qc.info["sfreq"],
                time_offset=start,
            )


class DynamicQCPipeline(BaseQCPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, raw: BrainsetsRaw) -> List[Optional[np.ndarray]]:
        return self.run(raw)

    def run(self, raw: BrainsetsRaw) -> np.ndarray:
        """
        Run the quality control pipeline on the input raw data.
        This method processes the raw EEG data by:
        1. Renaming channels (if specified)
        2. Setting the montage
        3. Splitting data into time slices
        4. Running QC pipeline on each slice

        Parameters
        ----------
        raw : BrainsetsRaw
            The raw EEG data to process. Must be an instance of BrainsetsRaw class.

        Returns
        -------
        List[np.ndarray]
            A list of quality scores for each time slice. Each element corresponds to
            the quality metrics for one time window. Failed slices return None.
        """
        # Rename channels
        if self.channels_rename is not None:
            raw.rename_channels(self.channels_rename)
            logging.info(f"Renamed channels: {self.channels_rename}.")

        # Set montage
        raw.set_montage(self.montage)

        # Detrend data
        # raw.detrend()

        # Create time slices
        time_slices = raw.slice(self.window_length)

        # Run QC pipeline and get quality scores
        scores = []
        for i, time_slice in enumerate(time_slices):
            start, end = time_slice

            logging.info(
                f"Processing time slice {i + 1} out of {len(time_slices)}: {start:.2f} to {end:.2f} seconds."
            )

            # try:
            # run preprocessing pipeline for single slice
            include_end = i == (len(time_slices) - 1)
            scores.append(
                self.run_slice(
                    raw.copy().crop(tmin=start, tmax=end, include_tmax=include_end)
                )
            )
            # except Exception as e:
            #     logging.error(
            #         f"Time slice {i + 1}: {start:.1f} to {end:.1f} seconds failed. \tError: {e}"
            #     )
            #     scores.append(None)

            # Log progress every N windows
            if (i + 1) % 5 == 0:
                logging.debug(f"Processed {i + 1}/{len(time_slices)} slices.")

        # Concatenate scores if more than one segment
        if len(scores) > 1:
            return np.concatenate(scores, axis=1)
        elif len(scores) == 1:
            return scores[0]
        return scores

    def run_slice(
        self,
        raw: BrainsetsRaw,
    ) -> Optional[np.ndarray]:
        """
        Process and evaluate the quality of EEG data for a single slice/segment.
        This method applies a series of preprocessing steps to the raw EEG data including:
            - High-pass filtering
            - Line noise removal
            - Bad channel detection and removal
            - Average referencing
            - Low-pass filtering
            - ICA fitting and application (optional)
            - Quality assessment using SPEED criteria (optional)
            - Computation of quality scores

        Parameters
        ----------
        raw : BrainsetsRaw
            Raw EEG data object to be processed.

        Returns
        -------
        numpy.ndarray or None
            Quality scores computed for each channel. Returns None if processing fails.

        Notes
        -----
        The method performs the following steps in order:
        1. Filters the data (high-pass and line noise removal)
        2. Detects and removes bad channels
        3. Applies average reference
        4. Applies low-pass filter
        5. Performs ICA if self.fit_ica is True
        6. Evaluates quality if self.quality_check is True
        7. Computes quality scores using RMSE
        The original raw data is copied before processing to preserve the input data.
        """
        # Make a copy of original Raw data
        raw_ = raw.copy()

        # High-pass filter to remove slow drifts
        raw_.bandpass_filter(
            hp_freq=self.hp_freq,
            lp_freq=None,
            verbose=False,
        )

        # Remove line noise
        raw_.remove_line_noise(line_freqs=self.line_freqs)

        # Drop bad channels
        raw_.detect_bad_channels(
            ransac=self.ransac,
            detrend=False,
            drop=True,
            return_bad=False,
            random_state=self.random_state,
            verbose=False,
        )

        # Average reference
        raw_.average_reference()

        # Low-pass filter for ICALabel
        raw_.bandpass_filter(
            hp_freq=None,
            lp_freq=self.lp_freq,
            verbose=False,
        )

        # Fit and apply ICA
        if self.fit_ica:
            # create a copy of preprocessed raw_ before ICA
            raw_before_ica = raw_.copy()

            # fit, evaluate and apply ICA
            self.ica = raw_.fit_ica(random_state=self.random_state, verbose=False)

            if self.quality_check_ica:
                quality_ica = raw_.evaluate_and_apply_ica(
                    self.ica,
                    self.iclabel_threshold,
                    self.include_components,
                    plot_excluded_components=False,
                    ignore_quality_score=True,
                    verbose=True,
                )

                logging.info(f"ICA QC passed: {quality_ica}")

            # remove bad channels after ICA
            raw_.detect_bad_channels(
                ransac=self.ransac,
                detrend=False,
                drop=True,
                return_bad=False,
                random_state=self.random_state,
                verbose=False,
            )

        # Evaluate quality based on SPEED criteria
        if self.quality_check:
            quality = self.speed_quality_asssessment(
                raw=raw_,
                verbose=True,
            )

            logging.info(f"General QC passed: {quality}")

        # Calculate quality scores
        scores = self.compute_quality_scores(
            score="rmse",
            raw=raw_before_ica,
            raw_qc=raw_,
            smooth=True,
            norm=True,
        )

        # Get scores across all channels
        scores = self.get_full_scores(raw=raw, raw_qc=raw_, scores=scores)

        if self.plot_scores:
            self.plot_quality_scores(
                raw=raw,
                raw_baseline=raw_before_ica,
                raw_qc=raw_,
                scores=scores,
            )

        return scores
