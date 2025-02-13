import io
import logging
from typing import Union

import numpy as np

import mne
from mne.io import RawArray
from mne.preprocessing import ICA
from mne_icalabel import label_components

import pyprep

import contextlib
from meegkit.detrend import detrend
from meegkit.dss import dss_line_iter

import matplotlib.pyplot as plt

from temporaldata import Data, Interval, LazyInterval


class BrainsetsRaw(RawArray):
    """A class to create a customized mne.io.RawArray-based object from
    a temporaldata.Data object.

    Args
        data : Data
            The custom Data object containing EEG data and metadata.
        verbose : bool, optional
            If True, enable verbose output. Default is False.

    """

    def __init__(self, data: Data, verbose: bool = False):
        # Sampling frequency
        sfreq = data.eeg.sampling_rate

        # Channel names and types
        ch_names = list(data.units.id)
        ch_types = ["eeg"] * len(ch_names)

        # EEG data
        signals = data.eeg.signal  # time x channels

        # Create info object
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=ch_types,
        )

        # Initializae mne.io.RawArray object
        super().__init__(
            data=signals.T,
            info=info,
            verbose=verbose,
        )

        # Standardize channel names
        self._to_standard_channel_names()

        # Create annotations object
        trials = data.imagined_movement_trials
        annot = mne.Annotations(
            onset=trials.start,
            duration=trials.end - trials.start,
            description=trials.movements,
        )
        super().set_annotations(annot)

        # Set intervals/epochs
        if "epochs" in data.keys():
            self.intervals = data.epochs
        else:
            self.intervals = None

    def _to_standard_channel_names(self):
        """Standardize EEG channel names according to the new 10-05 standard system."""
        # old -> new nomenclature
        rename_map = {"T3": "T7", "T4": "T8", "P7": "T5", "P8": "T6"}
        rename_map = {k: v for k, v in rename_map.items() if k in self.ch_names}

        super().rename_channels(rename_map)

    def set_montage(
        self,
        montage: mne.channels.DigMontage = None,
    ) -> None:
        """
        Set the montage for the EEG data.

        This method sets the montage (i.e., the electrode positions) for the EEG data.

        .. warning::
            Channels that are not included in the given montage will be dropped.

        Parameters
        ----------
        montage : mne.channels.DigMontage, optional
            The montage to set. If None, no montage will be set.
        """

        logging.warning(f"Channels not included in the given montage will be dropped.")

        drop_chs = [ch for ch in self.ch_names if ch not in montage.ch_names]
        super().drop_channels(drop_chs)

        if drop_chs:
            logging.warning(
                f"{len(drop_chs)} channels were dropped while setting montage ({drop_chs})."
            )
        else:
            logging.info(f"No channels were dropped while setting montage.")

        super().set_montage(montage)

        logging.info(f"Setting montage {montage}.")

    def slice(
        self, window_length: Union[int, float, None] = None
    ) -> list[tuple[float, float]]:
        """
        Creates time slices from Raw data according to specified criteria.

        This method provides three ways to slice the Raw data:
        1. Using fixed-length windows
        2. Using predefined intervals
        3. Using annotations

        Parameters
        ----------
        window_length : int or float, optional
            Length of each window in seconds. If provided, the Raw data will be sliced
            into fixed-length segments. If None, the method will attempt to use intervals
            or annotations.

        Returns
        -------
        list
            A list of tuples containing the start and end time point of each time slice.

        Raises
        ------
        RuntimeError
            If no window_length is provided and neither intervals nor annotations are available.

        Notes
        -----
        The slicing priority is:
        1. Window-based slicing (if window_length is provided)
        2. Interval-based slicing (if intervals exist and window_length is None)
        3. Annotation-based slicing (if annotations exist and window_length is None)
        """
        if isinstance(window_length, (int, float)):
            return self.slice_by_windows(window_length)
        elif (window_length is None) and self.intervals:
            return self.slice_by_intervals(self.intervals)
        elif (window_length is None) and self.annotations:
            return self.slice_by_annotations(self.annotations)
        else:
            logging.error(
                "No annotations or intervals found and no window length provided. Cannot slice Raw."
            )

    def slice_by_windows(
        self, window_length: Union[int, float] = 60, include_last: bool = True
    ) -> list[tuple[float, float]]:
        """
        Creates fixed-length time slices from Raw data.

        Parameters
        ----------
        window_length : int or float, optional
            Length of each window in seconds. Default is 60 seconds.
        include_last : bool, optional
            If True, include the last segment even if it is shorter than the window length. Default is True.

        Returns
        -------
        list of tuple
            A list of (start, end) tuples for each time slice.
        """
        time_slices = []
        start = self.times[0]
        total_duration = self.times[-1]
        while start < total_duration:
            end = start + window_length
            time_slices.append((start, min(end, total_duration)))
            # if end <= total_duration or include_last:
            #     time_slices.append((start, min(end, total_duration)))
            start += window_length

        logging.info(
            f"Slicing Raw into {len(time_slices)} ({window_length} sec.) windows."
            f"Total duration: {total_duration} sec. Last segment included: {include_last}."
        )

        return time_slices

    def slice_by_intervals(
        self, interval: Union[Interval, LazyInterval]
    ) -> list[tuple[float, float]]:
        """
        Creates time slices from Raw data based on Intervals.

        Parameters
        ----------
        interval : Interval
            An object containing 'start' and 'end' attributes, each representing arrays
            of time points defining the intervals.

        Returns
        -------
        list of tuple
            List of (start, end) time point tuples representing the slicing intervals.

        Notes
        -----
        The function creates time slices by pairing corresponding start and end points
        from the input Interval object. Each slice is represented as a tuple of
        (start_time, end_time).
        """
        total_duration = self.times[-1]
        time_slices = []
        for start, end in zip(interval.start, interval.end):
            time_slices.append((start, min(end, total_duration)))

        logging.info(f"Slicing Raw into {len(time_slices)} intervals.")

        return time_slices

    def slice_by_annotations(
        self,
        annotations: mne.annotations.Annotations,
    ) -> list[tuple[float, float]]:
        """
        Creates time slices from Raw data based on annotations.

        This method checks if the total duration of annotations matches the raw data duration,
        and creates a list of time slices (start, end) for each annotation.

        Parameters
        ----------
        annotations : mne.annotations.Annotations
            MNE Annotations object containing onset and duration information for each segment

        Returns
        -------
        list of tuple
            List of (onset, onset + duration) tuples defining time windows for each annotation.

        Raises
        ------
        ValueError
            If the total duration of annotations does not match the raw data duration
        """
        total_duration = self.times[-1]
        time_slices = []
        for onset, duration in zip(annotations.onset, annotations.duration):
            time_slices.append((onset, min(onset + duration, total_duration)))

        logging.info(f"Slicing Raw into {len(time_slices)} events.")

        return time_slices

    # preprocessing
    def detrend(self):
        """Performs detrending on the raw EEG data.
        This method applies polynomial detrending to remove linear trends from the data.
        If annotations are present, the data is split into segments based on annotation
        end points and detrending is applied separately to each segment before
        recombining.

        Notes
        -----
        - Uses MEEGkit's dss_line_iter implementation. A first order polynomial detrending (linear)
        - If annotations exist, detrending is done separately for each annotated segment
        - The detrended data is stored in-place in the raw object's _data attribute.
        """
        if self.annotations:
            # Get end time points for each annotation
            time_slices = self.slice_by_annotations()
            _, end = zip(*time_slices)

            # Convert end time points to samples
            end = np.array(end[:-1]) * int(self.info["sfreq"])

            # Split data into trials
            data_ = np.split(
                self.get_data(), indices_or_sections=end[:-1].astype(int), axis=1
            )
        else:
            data_ = [self.get_data()]

        # Detrend and transpose data
        def detrend_and_transpose(data):
            data, _, _ = detrend(
                data.T,
                order=1,
                basis="polynomials",
            )
            return data.T

        data_ = np.hstack([detrend_and_transpose(data) for data in data_])

        # assigned detrended data to the raw object
        self._data = data_

        logging.info(f"Detrending Raw data.")

    def bandpass_filter(
        self,
        hp_freq: float = None,
        lp_freq: float = None,
        verbose: bool = False,
    ) -> None:
        """Apply bandpass filter to the data using FIR method.

        Parameters
        ----------
        hp_freq : float, optional
            High-pass frequency in Hz. If None, no high-pass filter is applied.
        lp_freq : float, optional
            Low-pass frequency in Hz. If None, no low-pass filter is applied.
        verbose : bool, default False
            If True, show additional information about the filtering process.

        Notes
        -----
        The method checks if the low-pass frequency is above the Nyquist frequency
        (sampling_rate/2) and warns the user if this is the case, as filtering
        would not be effective.
        The filtering is performed using FIR (Finite Impulse Response) method.
        Depending on the parameters provided, the method will apply:
        - High-pass filter only (if only hp_freq is provided)
        - Low-pass filter only (if only lp_freq is provided)
        - Bandpass filter (if both frequencies are provided)
        - The filtering is applied in-place to the data.
        """
        sfreq = self.info["sfreq"]
        nyquist = sfreq / 2.0

        if (lp_freq is not None) and (lp_freq >= nyquist):
            logging.warning(
                f"Low-pass frequency {lp_freq} Hz is above or equal to the Nyquist frequency ({nyquist} Hz). Filtering will not be applied."
            )
        else:
            super().filter(
                l_freq=hp_freq, h_freq=lp_freq, method="fir", verbose=verbose
            )

            if (hp_freq is not None) and (lp_freq is None):
                logging.info(f"Applying high-pass filter at {hp_freq} Hz.")
            elif (hp_freq is None) and (lp_freq is not None):
                logging.info(f"Applying low-pass filter at {lp_freq} Hz.")
            elif (hp_freq is not None) and (lp_freq is not None):
                logging.info(
                    f"Applying bandpass filter in the {hp_freq} - {lp_freq} Hz frequency band."
                )

    def remove_line_noise(
        self,
        line_freqs: list,
    ) -> None:
        """Remove line noise from the data using DSS (Denoising Source Separation).
        This method applies an iterative DSS algorithm to remove power line noise at specified
        frequencies from the EEG data. It processes each line frequency separately as long
        as it's below the Nyquist frequency.

        Parameters
        ----------
        line_freqs : list
            List of line noise frequencies (in Hz) to remove from the data.
            Frequencies above Nyquist frequency will be ignored with a warning.

        Notes
        -----
        - Uses MEEGkit's dss_line_iter implementation
        - Silences stdout and stderr during processing
        - Logs info message for each processed frequency
        - Logs warning for frequencies above Nyquist
        - The method modifies the data in-place.
        """
        sfreq = self.info["sfreq"]
        nyquist = sfreq / 2.0

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            for line_freq in line_freqs:
                if line_freq < nyquist:
                    filtered_data, _ = dss_line_iter(
                        self.get_data().T,
                        fline=line_freq,
                        sfreq=sfreq,
                    )
                    self._data = filtered_data.T

                    logging.info(f"Removing line noise at {line_freq} Hz.")
                else:
                    logging.warning(
                        f"Line frequency {line_freq} is above the Nyquist frequency and will not be filtered."
                    )

    def detect_bad_channels(
        self,
        ransac: bool = True,
        detrend: bool = False,
        drop: bool = True,
        return_bad: bool = False,
        random_state: Union[None, int, np.random.RandomState] = None,
        verbose: bool = False,
    ) -> list:
        """Detect bad channels in the raw EEG data using the PREP pipeline.

        This method implements various criteria from the PREP pipeline to detect noisy channels,
        including:
            - Detection of NaN/flat channels
            - Detection of channels with abnormal standard deviation
            - Detection of channels with high frequency noise
            - Detection of poorly correlated channels
            - Detection of channels with poor SNR
            - Optional RANSAC-based detection

        Parameters
        ----------
        ransac : bool, default=True
            Whether to use RANSAC for bad channel detection. Only applies if number of
            channels >= 16.
        detrend : bool, default=False
            Whether to detrend the signal before bad channel detection.
        drop : bool, default=False
            If True, detected bad channels will be dropped from the data.
        return_bad : bool, default=False
            If True, returns the list of detected bad channels.
        random_state : None | int | numpy.random.RandomState, default=None
            Random state for RANSAC procedure.
        verbose : bool, default=False
            If True, prints detailed information about detected bad channels.

        Returns
        -------
        list
            List of bad channel names if return_bad=True, otherwise None.

        Notes
        -----
        If RANSAC is enabled but fails, the error will be logged but execution will continue.
        """
        noisychannels = pyprep.NoisyChannels(
            self,
            do_detrend=detrend,
            random_state=random_state,
        )

        noisychannels.find_bad_by_nan_flat()
        noisychannels.find_bad_by_deviation()
        noisychannels.find_bad_by_hfnoise()
        noisychannels.find_bad_by_correlation(frac_bad=0.05)
        noisychannels.find_bad_by_SNR()

        try:
            if ransac and self.info["nchan"] >= 16:
                noisychannels.find_bad_by_ransac()
        except Exception as e:
            logging.error(f"Error in RANSAC: {e}")

        bad_chs = noisychannels.get_bads()

        if verbose:
            logging.info(
                f"Number of bad channels detected: {len(bad_chs)} out of {self.info['nchan']} total channels"
            )
            _bad_channels = noisychannels.get_bads(as_dict=True)
            for key, value in _bad_channels.items():
                if value and key != "bad_all":
                    logging.info(f"\t{key}: {value}")

        if drop and bad_chs:
            super().drop_channels(bad_chs)
            logging.info(f"Dropping {len(bad_chs)} bad channels: {bad_chs}")

        if return_bad:
            return bad_chs

    def average_reference(self):
        """Apply average reference to EEG data.

        Average reference is implemented by subtracting the mean of all channels from each channel.
        This method modifies the Raw object in-place.

        Returns
        -------
        None
            The Raw object is modified in-place.

        Notes
        -----
        - This method uses MNE's set_eeg_reference function with 'average' reference.
        - The operation is performed without projection matrices and modifies the data directly.
        """
        mne.set_eeg_reference(
            self,
            ref_channels="average",
            projection=False,
            copy=False,
            verbose=False,
        )

        logging.info(f"Using average for referencing.")

    def fit_ica(
        self,
        random_state: Union[None, int, np.random.RandomState] = None,
        verbose=False,
    ) -> mne.preprocessing.ICA:
        """Fits an Independent Component Analysis (ICA) to the raw EEG data.
        This method implements the Infomax ICA algorithm with extended parameters, which is
        compatible with ICLabel. The number of components is automatically set to the matrix
        rank of the data.

        Parameters
        ----------
        random_state : Union[None, int, np.random.RandomState], optional
            Random state for reproducibility of the ICA decomposition, by default None.
        verbose : bool, optional
            If True, prints the explained variance ratio for the first 5 components,
            by default False.

        Returns
        -------
        mne.preprocessing.ICA
            The fitted ICA object that can be used for further analysis or component
            rejection.

        Notes
        -----
        The ICA is configured with:
        - Maximum 1000 iterations
        - Infomax method with extended parameters
        - Number of components equal to the matrix rank of the data
        If verbose is True, the method will log the percentage of EEG signal variance
        explained by each of the first 5 components.
        """

        # initialize ICA with the highest possible number of components
        # the Infomax methods is important for ICLabel
        ica = ICA(
            n_components=np.linalg.matrix_rank(self._data),
            max_iter=1000,
            method="infomax",
            fit_params=dict(extended=True),
            random_state=random_state,
            verbose=False,
        )

        ica.fit(self, verbose=False)

        logging.info(f"Fitting ICA to Raw.")

        if verbose:
            for component in range(5):
                explained_var_ratio = ica.get_explained_variance_ratio(
                    self, components=[component], ch_type="eeg"
                )
                ratio_percent = round(100 * explained_var_ratio["eeg"])

                logging.info(
                    f"Fraction of variance in EEG signal explained by component {component}: "
                    f"{ratio_percent}%"
                )

        return ica

    def evaluate_and_apply_ica(
        self,
        ica: mne.preprocessing.ICA,
        iclabel_threshold: float,
        include_components: list,
        ignore_quality_score: bool,
        plot_excluded_components: bool = False,
        verbose=False,
    ) -> bool:
        """
        Evaluate ICA components using ICLabel classification and apply ICA based on quality thresholds.

        This method applies ICLabel to classify ICA components, excludes components based on
        probability thresholds and component types, and optionally plots excluded components.
        The raw data is reconstructed after removing excluded components if quality criteria are met.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            Fitted ICA object to evaluate and apply
        iclabel_threshold : float
            Probability threshold above which components will be excluded if their label
            is not in include_components
        include_components : list
            List of ICLabel component types to keep regardless of probability
        ignore_quality_score : bool
            If True, applies ICA regardless of quality metrics
        plot_excluded_components : bool, default False
            If True, plots the excluded ICA components
        verbose : bool, default False
            If True, prints detailed information about excluded components

        Returns
        -------
        bool
            True if ICA passed quality threshold (ICR < 0.3), False otherwise

        Raises
        ------
        ValueError
            If ICA has not been fitted

        Notes
        -----
        The method uses an IC Rejection Ratio (ICR) threshold of 0.3, meaning if more than 30%
        of components are marked for rejection, the ICA solution is considered poor quality
        and will not be applied unless ignore_quality_score is True.
        """
        if ica is None:
            raise ValueError("ICA has not been fitted, cannot evaluate and apply.")

        # Run ICLabel
        ic_labels = label_components(self, ica, method="iclabel")
        labels = ic_labels["labels"]
        y_probs = ic_labels["y_pred_proba"]

        # Exclude components not in the include_components list and with high probability
        exclude_idx = [
            idx
            for idx, (label, y_prob) in enumerate(zip(labels, y_probs))
            if label not in include_components and y_prob > iclabel_threshold
        ]

        if plot_excluded_components:
            # ica.plot_sources(raw, show_scrollbars=False)
            # plt.show()

            ica.plot_components(picks=exclude_idx)
            plt.show()

            for i, (label, y_prob) in enumerate(zip(labels, y_probs)):
                if i in exclude_idx:
                    logging.info(
                        f"Excluding component {i}: {label} with {y_prob * 100:.0f}% prob."
                    )

        # Estimate quality score
        icr = len(exclude_idx) / len(labels)
        icr_threshold = 0.3

        if (icr < icr_threshold) or ignore_quality_score:
            # Reconstruct the raw data
            self = ica.apply(self, exclude=exclude_idx, verbose=False)

            if verbose:
                logging.info(
                    f"ICA-QC: excluding {len(exclude_idx)} out of {len(labels)} components ({icr * 100:.1f}%)."
                )
                if not plot_excluded_components:
                    for i, (label, y_prob) in enumerate(zip(labels, y_probs)):
                        if i in exclude_idx:
                            logging.info(
                                f"Excluding component {i}: {label} with {y_prob * 100:.1f}% prob."
                            )

        return icr < icr_threshold
