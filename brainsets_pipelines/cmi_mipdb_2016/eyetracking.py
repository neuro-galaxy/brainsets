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


def _parse_user_events_from_file(path: Path) -> list[tuple[int, str]]:
    """Read all UserEvent lines from an ET file (Events or Samples).

    Returns list of (timestamp_µs, description) tuples.
    """
    user_events = []
    with open(path, errors="replace") as f:
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


def _et_file_base(path: Path) -> str:
    """Return the base name of an ET file (without ' Events.txt' or ' Samples.txt')."""
    name = path.name
    for suffix in (" Events.txt", " Samples.txt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def build_et_index(et_dir: Path) -> list[dict]:
    """Build a unified index of all eyetracking data from all files (Samples and Events).

    Returns a list of dicts, one per ET "base" (logical file pair). Each dict has:
    - ``base_id``: str
    - ``samples_path``: Path | None
    - ``events_path``: Path | None
    - ``segments``: list of segment dicts (code, paradigm_ts, start_ts, end_ts)
    """
    if not et_dir or not et_dir.exists():
        return []

    bases: dict[str, dict[str, Path | None]] = {}
    for path in et_dir.glob("* Samples.txt"):
        base = _et_file_base(path)
        bases.setdefault(base, {"samples_path": None, "events_path": None})
        bases[base]["samples_path"] = path
    for path in et_dir.glob("* Events.txt"):
        base = _et_file_base(path)
        bases.setdefault(base, {"samples_path": None, "events_path": None})
        bases[base]["events_path"] = path

    index: list[dict] = []
    for base_id in sorted(bases.keys()):
        entry = bases[base_id]
        samples_path = entry["samples_path"]
        events_path = entry["events_path"]

        if events_path is not None and events_path.exists():
            user_events = _parse_user_events_from_file(events_path)
        elif samples_path is not None and samples_path.exists():
            user_events = _parse_user_events_from_file(samples_path)
        else:
            continue

        if not user_events:
            continue

        segments = _identify_segments(user_events)
        index.append(
            {
                "base_id": base_id,
                "samples_path": (
                    samples_path if samples_path and samples_path.exists() else None
                ),
                "events_path": (
                    events_path if events_path and events_path.exists() else None
                ),
                "segments": segments,
            }
        )

    return index


def find_paradigm_segment(
    et_dir: Path,
    paradigm_code: int,
    index: list[dict] | None = None,
) -> dict | None:
    """Find an ET segment for *paradigm_code* from the unified ET index.

    Returns a dict with keys ``events_path`` (optional), ``samples_path`` (optional),
    ``code``, ``paradigm_ts``, ``start_ts``, ``end_ts``. Paths are None when that
    file is not present for this segment.
    """
    if index is None:
        index = build_et_index(et_dir)

    for entry in index:
        samples_path = entry["samples_path"]
        events_path = entry["events_path"]
        for seg in entry["segments"]:
            if seg["code"] == paradigm_code:
                return {
                    "events_path": events_path,
                    "samples_path": samples_path,
                    **seg,
                }

    return None


def _convert_timestamp(
    t_us: int | np.ndarray,
    paradigm_ts_us: int,
    eeg_paradigm_onset_s: float,
) -> float | np.ndarray:
    """Map ET microsecond timestamp(s) to EEG-aligned seconds."""
    return (
        np.asarray(t_us, dtype=np.float64) - paradigm_ts_us
    ) / 1_000_000.0 + eeg_paradigm_onset_s


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


def process(
    et_dir: Path,
    paradigm_code: int,
    eeg_paradigm_onset_s: float,
) -> Data | None:
    """Build an eyetracking ``Data`` for a single paradigm from the unified ET index.

    Returns a ``Data`` with up to four ``IrregularTimeSeries``
    (``samples``, ``fixations``, ``saccades``, ``blinks``), or ``None``.
    """
    index = build_et_index(et_dir)
    segment = find_paradigm_segment(et_dir, paradigm_code, index=index)
    if segment is None:
        return None

    paradigm_ts = segment["paradigm_ts"]
    start_ts = segment["start_ts"]
    end_ts = segment["end_ts"]

    et_kwargs: dict = {}

    if segment.get("samples_path") is not None:
        samples = parse_samples(
            segment["samples_path"],
            start_ts,
            end_ts,
            paradigm_ts,
            eeg_paradigm_onset_s,
        )
        if samples is not None:
            et_kwargs["samples"] = samples

    if segment.get("events_path") is not None:
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
    return Data(domain="auto", **et_kwargs)
