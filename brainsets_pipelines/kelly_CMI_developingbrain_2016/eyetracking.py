"""Parsing and processing of SMI iView eyetracking text exports.

The public entry point is :func:`process`, which returns a
``temporaldata.Data`` object containing up to four
``IrregularTimeSeries``: *samples*, *fixations*, *saccades*, *blinks*.
All timestamps are aligned to the EEG time base via the shared paradigm onset marker.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from temporaldata import Data, IrregularTimeSeries

from constants import PARADIGM_MAP, SAMPLES_COLUMNS


def _parse_user_events(events_path: Path) -> list[tuple[int, str]]:
    """Read all UserEvent lines from an ET Events file.

    Returns list of (timestamp_µs, description) tuples.
    """
    user_events = []
    with open(events_path, errors="replace") as f:
        for line in f:
            if line.startswith("UserEvent"):
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    try:
                        ts = int(parts[3])
                    except ValueError:
                        continue
                    user_events.append((ts, parts[4].strip()))
    return user_events


def _identify_segments(user_events: list[tuple[int, str]]) -> list[dict]:
    """Identify paradigm segments bounded by ``start_eye_recording`` markers.

    Returns list of dicts with keys: code, paradigm_ts, start_ts, end_ts.
    ``end_ts`` is ``None`` for the last segment.
    """
    recording_starts = [ts for ts, desc in user_events if "start_eye_recording" in desc]

    segments = []
    for ts, desc in user_events:
        m = re.match(r"# Message: (\d+)$", desc)
        if not m:
            continue
        code = int(m.group(1))
        if code not in PARADIGM_MAP:
            continue

        preceding_start = None
        for rs in recording_starts:
            if rs < ts:
                preceding_start = rs
        following_start = None
        for rs in recording_starts:
            if rs > ts:
                following_start = rs
                break

        segments.append(
            {
                "code": code,
                "paradigm_ts": ts,
                "start_ts": (preceding_start if preceding_start else user_events[0][0]),
                "end_ts": following_start,
            }
        )
    return segments


def _find_resting_in_vis_learn(et_dir: Path) -> dict | None:
    """Locate the resting-paradigm ET segment in ``*_vis_learn`` files.

    The eye-tracking data for the resting paradigm are stored in the eye-tracking
    data files of the Sequence Learning paradigm (code 90).
    """
    for events_path in sorted(et_dir.glob("*vis_learn Events.txt")):
        samples_path = events_path.parent / events_path.name.replace(
            " Events.txt", " Samples.txt"
        )
        if not samples_path.exists():
            continue

        user_events = _parse_user_events(events_path)
        recording_starts = [
            ts for ts, desc in user_events if "start_eye_recording" in desc
        ]

        if len(recording_starts) >= 2:
            return {
                "events_path": events_path,
                "samples_path": samples_path,
                "code": 90,
                "paradigm_ts": recording_starts[0],
                "start_ts": recording_starts[0],
                "end_ts": recording_starts[1],
            }
    return None


def find_paradigm_segment(et_dir: Path, paradigm_code: int) -> dict | None:
    """Scan ET Events files in *et_dir* for a segment matching *paradigm_code*.

    Returns a dict with keys ``events_path``, ``samples_path``, ``code``,
    ``paradigm_ts``, ``start_ts``, ``end_ts`` — or ``None``.
    """
    if not et_dir or not et_dir.exists():
        return None

    events_files = sorted(et_dir.glob("* Events.txt"))

    for events_path in events_files:
        samples_path = events_path.parent / events_path.name.replace(
            " Events.txt", " Samples.txt"
        )
        if not samples_path.exists():
            logging.warning(f"Missing Samples file for {events_path.name}")
            continue

        user_events = _parse_user_events(events_path)
        segments = _identify_segments(user_events)

        for seg in segments:
            if seg["code"] == paradigm_code:
                return {
                    "events_path": events_path,
                    "samples_path": samples_path,
                    **seg,
                }

    if paradigm_code == 90:
        return _find_resting_in_vis_learn(et_dir)

    return None


# ======================================================================
# Timestamp conversion
# ======================================================================


def _convert_timestamp(
    t_us: int | np.ndarray,
    paradigm_ts_us: int,
    eeg_paradigm_onset_s: float,
) -> float | np.ndarray:
    """Map ET microsecond timestamp(s) to EEG-aligned seconds."""
    return (
        np.asarray(t_us, dtype=np.float64) - paradigm_ts_us
    ) / 1_000_000.0 + eeg_paradigm_onset_s


# ======================================================================
# File parsers
# ======================================================================


def parse_samples(
    samples_path: Path,
    start_ts: int,
    end_ts: int | None,
    paradigm_ts: int,
    eeg_paradigm_onset_s: float,
) -> IrregularTimeSeries | None:
    """Parse SMP rows from a Samples.txt file within *[start_ts, end_ts)*."""
    header_idx = None
    with open(samples_path, errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("Time\t"):
                header_idx = i
                break
    if header_idx is None:
        logging.warning(f"No column header found in {samples_path}")
        return None

    df = pd.read_csv(
        samples_path,
        sep="\t",
        skiprows=header_idx,
        low_memory=False,
        on_bad_lines="skip",
    )

    smp_df = df[df["Type"] == "SMP"].copy()
    if smp_df.empty:
        return None

    smp_df["Time"] = pd.to_numeric(smp_df["Time"], errors="coerce")
    smp_df = smp_df.dropna(subset=["Time"])
    smp_df["Time"] = smp_df["Time"].astype(np.int64)

    mask = smp_df["Time"] >= start_ts
    if end_ts is not None:
        mask &= smp_df["Time"] < end_ts
    smp_df = smp_df[mask]

    if smp_df.empty:
        return None

    timestamps = _convert_timestamp(
        smp_df["Time"].values, paradigm_ts, eeg_paradigm_onset_s
    )

    kwargs: dict = {}
    for src_col, dst_col in SAMPLES_COLUMNS.items():
        if src_col in smp_df.columns:
            kwargs[dst_col] = pd.to_numeric(
                smp_df[src_col], errors="coerce"
            ).values.astype(np.float32)

    return IrregularTimeSeries(
        timestamps=timestamps.astype(np.float64),
        domain="auto",
        **kwargs,
    )


def parse_events(
    events_path: Path,
    start_ts: int,
    end_ts: int | None,
    paradigm_ts: int,
    eeg_paradigm_onset_s: float,
) -> dict:
    """Parse fixation / saccade / blink rows from an Events.txt file.

    Returns a dict whose values are ``IrregularTimeSeries`` keyed by
    ``"fixations"``, ``"saccades"``, ``"blinks"`` (only present keys).
    """
    fixations: list[dict] = []
    saccades: list[dict] = []
    blinks: list[dict] = []

    with open(events_path, errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            etype = parts[0]

            try:
                if etype.startswith("Fixation") and len(parts) >= 13:
                    ev_ts = int(parts[3])
                    if ev_ts < start_ts or (end_ts and ev_ts >= end_ts):
                        continue
                    fixations.append(
                        {
                            "ts": ev_ts,
                            "eye": etype.split()[-1],
                            "duration": int(parts[5]),
                            "location_x": float(parts[6]),
                            "location_y": float(parts[7]),
                            "dispersion_x": float(parts[8]),
                            "dispersion_y": float(parts[9]),
                            "avg_pupil_size_x": float(parts[11]),
                            "avg_pupil_size_y": float(parts[12]),
                        }
                    )
                elif etype.startswith("Saccade") and len(parts) >= 14:
                    ev_ts = int(parts[3])
                    if ev_ts < start_ts or (end_ts and ev_ts >= end_ts):
                        continue
                    saccades.append(
                        {
                            "ts": ev_ts,
                            "eye": etype.split()[-1],
                            "duration": int(parts[5]),
                            "start_x": float(parts[6]),
                            "start_y": float(parts[7]),
                            "end_x": float(parts[8]),
                            "end_y": float(parts[9]),
                            "amplitude": float(parts[10]),
                            "peak_speed": float(parts[11]),
                            "avg_speed": float(parts[13]),
                        }
                    )
                elif etype.startswith("Blink") and len(parts) >= 6:
                    ev_ts = int(parts[3])
                    if ev_ts < start_ts or (end_ts and ev_ts >= end_ts):
                        continue
                    blinks.append(
                        {
                            "ts": ev_ts,
                            "eye": etype.split()[-1],
                            "duration": int(parts[5]),
                        }
                    )
            except (ValueError, IndexError):
                continue

    result: dict = {}

    if fixations:
        raw_ts = np.array([f["ts"] for f in fixations])
        result["fixations"] = IrregularTimeSeries(
            timestamps=_convert_timestamp(raw_ts, paradigm_ts, eeg_paradigm_onset_s),
            domain="auto",
            eye=np.array([f["eye"] for f in fixations], dtype="U1"),
            duration=np.array([f["duration"] for f in fixations], dtype=np.float64)
            / 1e6,
            location_x=np.array([f["location_x"] for f in fixations], dtype=np.float32),
            location_y=np.array([f["location_y"] for f in fixations], dtype=np.float32),
            dispersion_x=np.array(
                [f["dispersion_x"] for f in fixations], dtype=np.float32
            ),
            dispersion_y=np.array(
                [f["dispersion_y"] for f in fixations], dtype=np.float32
            ),
            avg_pupil_size_x=np.array(
                [f["avg_pupil_size_x"] for f in fixations], dtype=np.float32
            ),
            avg_pupil_size_y=np.array(
                [f["avg_pupil_size_y"] for f in fixations], dtype=np.float32
            ),
        )

    if saccades:
        raw_ts = np.array([s["ts"] for s in saccades])
        result["saccades"] = IrregularTimeSeries(
            timestamps=_convert_timestamp(raw_ts, paradigm_ts, eeg_paradigm_onset_s),
            domain="auto",
            eye=np.array([s["eye"] for s in saccades], dtype="U1"),
            duration=np.array([s["duration"] for s in saccades], dtype=np.float64)
            / 1e6,
            start_x=np.array([s["start_x"] for s in saccades], dtype=np.float32),
            start_y=np.array([s["start_y"] for s in saccades], dtype=np.float32),
            end_x=np.array([s["end_x"] for s in saccades], dtype=np.float32),
            end_y=np.array([s["end_y"] for s in saccades], dtype=np.float32),
            amplitude=np.array([s["amplitude"] for s in saccades], dtype=np.float32),
            peak_speed=np.array([s["peak_speed"] for s in saccades], dtype=np.float32),
            avg_speed=np.array([s["avg_speed"] for s in saccades], dtype=np.float32),
        )

    if blinks:
        raw_ts = np.array([b["ts"] for b in blinks])
        result["blinks"] = IrregularTimeSeries(
            timestamps=_convert_timestamp(raw_ts, paradigm_ts, eeg_paradigm_onset_s),
            domain="auto",
            eye=np.array([b["eye"] for b in blinks], dtype="U1"),
            duration=np.array([b["duration"] for b in blinks], dtype=np.float64) / 1e6,
        )

    return result


# ======================================================================
# Public entry point
# ======================================================================


def process(
    et_dir: Path,
    paradigm_code: int,
    eeg_paradigm_onset_s: float,
) -> Data | None:
    """Build an eyetracking ``Data`` for a single paradigm.

    Returns a ``Data`` with up to four ``IrregularTimeSeries``
    (``samples``, ``fixations``, ``saccades``, ``blinks``), all
    time-aligned to the EEG recording, or ``None`` if no matching
    ET data is found.
    """
    segment = find_paradigm_segment(et_dir, paradigm_code)
    if segment is None:
        return None

    paradigm_ts = segment["paradigm_ts"]
    start_ts = segment["start_ts"]
    end_ts = segment["end_ts"]

    et_kwargs: dict = {}

    samples = parse_samples(
        segment["samples_path"],
        start_ts,
        end_ts,
        paradigm_ts,
        eeg_paradigm_onset_s,
    )
    if samples is not None:
        et_kwargs["samples"] = samples

    events = parse_events(
        segment["events_path"],
        start_ts,
        end_ts,
        paradigm_ts,
        eeg_paradigm_onset_s,
    )
    et_kwargs.update(events)

    if not et_kwargs:
        return None
    return Data(**et_kwargs)
