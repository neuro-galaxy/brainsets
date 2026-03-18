"""Parsing and processing of SMI iView eyetracking text exports.

The public entry point is :func:`process`, which returns a
``temporaldata.Data`` object containing up to four
``IrregularTimeSeries``: *samples*, *fixations*, *saccades*, *blinks*.
All timestamps are aligned to the EEG time base via the shared paradigm onset marker.

Eye tracking data are based on two separate files:
- Samples.txt: contains eye tracking data (Dia, POR, pupil dilation, etc.).
- Events.txt: contains annotations (Fixation, Saccade, Blink).

Not all EEG paradigms have simultaneous Eye Tracking data.
Some Eye tracking recordings contain multiple paradigms. 
Matching of EEG and Eye tracking data is done by matching the paradigm codes
according to the UserEvent messages in the eye tracking Samples.txt files.

We build a single unified index from all ET files (Samples and Events), then
identify paradigms and parse from that index. See :func:`build_et_index` and
:func:`find_paradigm_segment`.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from temporaldata import Data, Interval, IrregularTimeSeries

from constants import PARADIGM_MAP, SAMPLES_COLUMNS


def _parse_user_events_from_file(path: Path) -> list[tuple[int, str]]:
    """Read all UserEvent lines from an ET file (Events or Samples).

    UserEvent lines are used to match ET data to EEG paradigms.

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

    Returns list of dicts with keys: code, paradigm_ts, start_ts, end_ts, all_events.
    ``end_ts`` is ``None`` for the last segment.
    ``all_events`` is a list of ``(timestamp_us, code)`` for every numeric
    ``# Message: <int>`` UserEvent within ``[start_ts, end_ts)``.
    """
    recording_starts = [ts for ts, desc in user_events if "start_eye_recording" in desc]

    numeric_events: list[tuple[int, int]] = []
    for ts, desc in user_events:
        m = re.match(r"# Message: (\d+)$", desc)
        if m:
            numeric_events.append((ts, int(m.group(1))))

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

        start_ts = preceding_start if preceding_start else user_events[0][0]
        end_ts = following_start

        seg_events = [
            (ev_ts, ev_code)
            for ev_ts, ev_code in numeric_events
            if ev_ts >= start_ts and (end_ts is None or ev_ts < end_ts)
        ]

        segments.append(
            {
                "code": code,
                "paradigm_ts": ts,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "all_events": seg_events,
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

    Scans all ET data files and groups them.
    Reads UserEvents from each group and identifies all paradigm segments.
    Returns an index with all paradigm segments found.

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


def _find_lcs(seq_a: list[int], seq_b: list[int]) -> list[tuple[int, int]]:
    """Return index pairs for one Longest Common Subsequence of *seq_a* and *seq_b*."""
    n, m = len(seq_a), len(seq_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for r in range(n - 1, -1, -1):
        val = seq_a[r]
        for c in range(m - 1, -1, -1):
            if val == seq_b[c]:
                dp[r][c] = dp[r + 1][c + 1] + 1
            else:
                dp[r][c] = max(dp[r + 1][c], dp[r][c + 1])

    r, c = 0, 0
    pairs: list[tuple[int, int]] = []
    while r < n and c < m:
        if seq_a[r] == seq_b[c]:
            pairs.append((r, c))
            r += 1
            c += 1
        elif dp[r + 1][c] >= dp[r][c + 1]:
            r += 1
        else:
            c += 1
    return pairs


def match_eeg_to_et(
    paradigm_intervals: Interval,
    et_index: list[dict],
) -> dict | None:
    """Match an EEG session to the best ET file using LCS on paradigm codes.

    Returns ``None`` if no match, or a dict with:
        ``base_id``, ``samples_path``, ``events_path`` — the matched ET file,
        ``segment_map`` — ``{eeg_paradigm_idx: segment_dict}`` mapping.
    """
    if len(paradigm_intervals) == 0 or not et_index:
        return None

    eeg_codes = [int(c) for c in paradigm_intervals.code]

    best_entry = None
    best_pairs: list[tuple[int, int]] = []

    for entry in et_index:
        et_codes = [int(seg["code"]) for seg in entry["segments"]]
        pairs = _find_lcs(eeg_codes, et_codes)
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_entry = entry

    if not best_pairs or best_entry is None:
        return None

    segment_map = {}
    for eeg_idx, et_idx in best_pairs:
        segment_map[eeg_idx] = best_entry["segments"][et_idx]

    return {
        "base_id": best_entry["base_id"],
        "samples_path": best_entry["samples_path"],
        "events_path": best_entry["events_path"],
        "segment_map": segment_map,
    }


def _align_paradigm_events(
    eeg_events: list[tuple[float, int]],
    et_events: list[tuple[int, int]],
) -> list[tuple[float, int]]:
    """Match EEG and ET event codes within one paradigm via LCS.

    Returns a list of ``(eeg_timestamp_s, et_timestamp_us)`` anchor pairs.
    """
    eeg_codes = [code for _, code in eeg_events]
    et_codes = [code for _, code in et_events]
    pairs = _find_lcs(eeg_codes, et_codes)
    return [(eeg_events[ei][0], et_events[ti][0]) for ei, ti in pairs]


def _eeg_events_in_interval(
    annotations: Interval, start_s: float, end_s: float
) -> list[tuple[float, int]]:
    """Slice ``annotations`` to ``[start_s, end_s)`` and return ``(timestamp_s, code)`` pairs."""
    mask = (annotations.start >= start_s) & (annotations.start < end_s)
    events: list[tuple[float, int]] = []
    for idx in np.where(mask)[0]:
        try:
            code = int(float(str(annotations.description[idx]).strip()))
        except (ValueError, TypeError):
            continue
        events.append((float(annotations.start[idx]), code))
    return events


def compute_et_to_eeg_time_transform(
    matched: dict,
    annotations: Interval,
    paradigm_intervals: Interval,
) -> tuple[float, float] | None:
    """Fit a linear ET-to-EEG timestamp transform from all matched paradigms.
    Finds the relationship between the ET and EEG timestamps for a given paradigm.

    Returns ``(slope, intercept)`` or ``None`` if no anchors are found.
    """
    all_anchors: list[tuple[float, int]] = []
    for paradigm_idx, seg in matched["segment_map"].items():
        eeg_events = _eeg_events_in_interval(
            annotations,
            float(paradigm_intervals.start[paradigm_idx]),
            float(paradigm_intervals.end[paradigm_idx]),
        )
        et_events = seg.get("all_events", [])
        anchors = _align_paradigm_events(eeg_events, et_events)
        if not anchors:
            eeg_start = float(paradigm_intervals.start[paradigm_idx])
            et_start = seg.get("paradigm_ts")
            if et_start is not None:
                anchors = [(eeg_start, int(et_start))]
                logging.warning(
                    "No sub-event matches for paradigm code %s; "
                    "using paradigm-start anchor only.",
                    seg.get("code"),
                )
        all_anchors.extend(anchors)

    if not all_anchors:
        return None

    et_s = np.array([a[1] for a in all_anchors], dtype=np.float64) / 1_000_000.0
    eeg_s = np.array([a[0] for a in all_anchors], dtype=np.float64)

    if len(all_anchors) == 1:
        slope = 1.0
        intercept = float(eeg_s[0] - et_s[0])
    else:
        slope, intercept = np.polyfit(et_s, eeg_s, 1)
        slope = float(slope)
        intercept = float(intercept)
        residuals = slope * et_s + intercept - eeg_s
        rmse = float(np.sqrt(np.mean(residuals**2)))
        logging.info(
            "Matching EEG and ET timescales",
            len(all_anchors),
            slope,
            intercept,
            rmse,
        )

    return (slope, intercept)


def find_paradigm_segment(
    et_dir: Path,
    paradigm_code: int,
    index: list[dict] | None = None,
) -> dict | None:
    """Find an ET segment for *paradigm_code* from the unified ET index.
    If no index is provided, builds the index from `et_dir` and searches it.

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


def _et_to_eeg_timestamps(
    t_us: int | np.ndarray, slope: float, intercept: float
) -> np.ndarray:
    """Convert ET microsecond timestamps to EEG seconds."""
    return slope * np.asarray(t_us, dtype=np.float64) / 1_000_000.0 + intercept


def parse_samples(
    samples_path: Path,
    slope: float,
    intercept: float,
    eeg_start_s: float,
    eeg_end_s: float,
) -> IrregularTimeSeries | None:
    """Parse SMP rows from a Samples.txt file, convert to EEG seconds, clip to EEG domain."""
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

    timestamps = _et_to_eeg_timestamps(smp_df["Time"].values, slope, intercept)

    keep = (timestamps >= eeg_start_s) & (timestamps <= eeg_end_s)
    smp_df = smp_df[keep]
    timestamps = timestamps[keep]

    if timestamps.size == 0:
        return None

    kwargs: dict = {}
    for src_col, dst_col in SAMPLES_COLUMNS.items():
        if src_col in smp_df.columns:
            kwargs[dst_col] = pd.to_numeric(
                smp_df[src_col], errors="coerce"
            ).values.astype(np.float32)

    return IrregularTimeSeries(
        timestamps=timestamps,
        domain="auto",
        **kwargs,
    )


def parse_events(
    events_path: Path,
    slope: float,
    intercept: float,
    eeg_start_s: float,
    eeg_end_s: float,
) -> dict:
    """Parse fixation / saccade / blink rows from an Events.txt file.

    Converts all timestamps to EEG seconds and clips to ``[eeg_start_s, eeg_end_s]``.
    Returns a dict whose values are ``IrregularTimeSeries`` keyed by
    ``"fixations"``, ``"saccades"``, ``"blinks"`` (only non-empty keys).
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
                    fixations.append(
                        {
                            "ts": int(parts[3]),
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
                    saccades.append(
                        {
                            "ts": int(parts[3]),
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
                    blinks.append(
                        {
                            "ts": int(parts[3]),
                            "eye": etype.split()[-1],
                            "duration": int(parts[5]),
                        }
                    )
            except (ValueError, IndexError):
                continue

    result: dict = {}

    if fixations:
        raw_ts = np.array([f["ts"] for f in fixations])
        eeg_ts = _et_to_eeg_timestamps(raw_ts, slope, intercept)
        keep = (eeg_ts >= eeg_start_s) & (eeg_ts <= eeg_end_s)
        if np.any(keep):
            kept = [fixations[i] for i in np.where(keep)[0]]
            result["fixations"] = IrregularTimeSeries(
                timestamps=eeg_ts[keep],
                domain="auto",
                eye=np.array([f["eye"] for f in kept], dtype="U1"),
                duration=np.array([f["duration"] for f in kept], dtype=np.float64)
                / 1e6,
                location_x=np.array([f["location_x"] for f in kept], dtype=np.float32),
                location_y=np.array([f["location_y"] for f in kept], dtype=np.float32),
                dispersion_x=np.array(
                    [f["dispersion_x"] for f in kept], dtype=np.float32
                ),
                dispersion_y=np.array(
                    [f["dispersion_y"] for f in kept], dtype=np.float32
                ),
                avg_pupil_size_x=np.array(
                    [f["avg_pupil_size_x"] for f in kept], dtype=np.float32
                ),
                avg_pupil_size_y=np.array(
                    [f["avg_pupil_size_y"] for f in kept], dtype=np.float32
                ),
            )

    if saccades:
        raw_ts = np.array([s["ts"] for s in saccades])
        eeg_ts = _et_to_eeg_timestamps(raw_ts, slope, intercept)
        keep = (eeg_ts >= eeg_start_s) & (eeg_ts <= eeg_end_s)
        if np.any(keep):
            kept = [saccades[i] for i in np.where(keep)[0]]
            result["saccades"] = IrregularTimeSeries(
                timestamps=eeg_ts[keep],
                domain="auto",
                eye=np.array([s["eye"] for s in kept], dtype="U1"),
                duration=np.array([s["duration"] for s in kept], dtype=np.float64)
                / 1e6,
                start_x=np.array([s["start_x"] for s in kept], dtype=np.float32),
                start_y=np.array([s["start_y"] for s in kept], dtype=np.float32),
                end_x=np.array([s["end_x"] for s in kept], dtype=np.float32),
                end_y=np.array([s["end_y"] for s in kept], dtype=np.float32),
                amplitude=np.array([s["amplitude"] for s in kept], dtype=np.float32),
                peak_speed=np.array([s["peak_speed"] for s in kept], dtype=np.float32),
                avg_speed=np.array([s["avg_speed"] for s in kept], dtype=np.float32),
            )

    if blinks:
        raw_ts = np.array([b["ts"] for b in blinks])
        eeg_ts = _et_to_eeg_timestamps(raw_ts, slope, intercept)
        keep = (eeg_ts >= eeg_start_s) & (eeg_ts <= eeg_end_s)
        if np.any(keep):
            kept = [blinks[i] for i in np.where(keep)[0]]
            result["blinks"] = IrregularTimeSeries(
                timestamps=eeg_ts[keep],
                domain="auto",
                eye=np.array([b["eye"] for b in kept], dtype="U1"),
                duration=np.array([b["duration"] for b in kept], dtype=np.float64)
                / 1e6,
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
