"""Extraction and mapping of paradigm intervals from EEG annotations."""

import numpy as np
from temporaldata import Interval

from brainsets.taxonomy import Task

from constants import PARADIGM_END_CODES, PARADIGM_MAP


def _annotation_code_at(annotations: Interval, i: int) -> int | None:
    """Return the annotation code at index i, or None if not parseable."""
    if annotations.description is None or i >= len(annotations.description):
        return None
    try:
        return int(float(str(annotations.description[i]).strip()))
    except (ValueError, TypeError):
        return None


def _paradigm_segment_end(
    start_idx: int,
    paradigm_code: int,
    annotations: Interval,
    paradigm_start_indices: list[int],
) -> float:
    """Return the end time for a paradigm segment that starts at start_idx.

    - For NATURALISTIC_VIEWING: end is the time of the first annotation after
      start whose code is in PARADIGM_END_CODES for this paradigm (use that
      annotation's end time). If none found, use the last annotation's end.
    - For other paradigms: end is the end time of the last annotation before
      the next paradigm start, or the last annotation in the recording if
      there is no next paradigm.
    """
    n = len(annotations.start)
    if paradigm_code not in PARADIGM_MAP:
        return float(annotations.end[start_idx])

    _, task = PARADIGM_MAP[paradigm_code]
    if task == Task.NATURALISTIC_VIEWING:
        end_codes = PARADIGM_END_CODES.get(paradigm_code, [])
        for j in range(start_idx + 1, n):
            code_j = _annotation_code_at(annotations, j)
            if code_j is not None and code_j in end_codes:
                return float(annotations.end[j])
        return (
            float(annotations.end[-1]) if n > 0 else float(annotations.end[start_idx])
        )

    # Last annotation before next paradigm start, or last in recording
    try:
        pos = paradigm_start_indices.index(start_idx)
        next_idx = (
            paradigm_start_indices[pos + 1]
            if pos + 1 < len(paradigm_start_indices)
            else None
        )
    except ValueError:
        next_idx = None
    if next_idx is not None and next_idx > 0:
        return float(annotations.end[next_idx - 1])
    return float(annotations.end[-1]) if n > 0 else float(annotations.end[start_idx])


def _extract_event_codes(
    annotations: Interval, start_s: float, end_s: float
) -> np.ndarray:
    """Return all numeric annotation codes within ``[start_s, end_s]``."""
    mask = (annotations.start >= start_s) & (annotations.start <= end_s)
    codes: list[int] = []
    for idx in np.where(mask)[0]:
        code = _annotation_code_at(annotations, idx)
        if code is not None:
            codes.append(code)
    return np.array(codes, dtype=np.int64)


def get_paradigm_interval_for_code(
    paradigm_code: int,
    annotations: Interval,
) -> Interval:
    """Return the interval domain(s) for a specific paradigm code in the recording.

    Args:
        paradigm_code: Paradigm start code (must be in PARADIGM_MAP).
        annotations: Raw annotations from the recording (start, end, description).

    Returns:
        Interval with one segment per occurrence of this paradigm (start, end,
        description, code, event_codes). Empty Interval if the paradigm does
        not appear.
    """
    if paradigm_code not in PARADIGM_MAP or len(annotations) == 0:
        return Interval(start=np.array([]), end=np.array([]))
    if annotations.description is None:
        return Interval(start=np.array([]), end=np.array([]))

    n = len(annotations.start)
    paradigm_start_indices = [
        i for i in range(n) if _annotation_code_at(annotations, i) in PARADIGM_MAP
    ]
    indices_for_code = [
        i
        for i in paradigm_start_indices
        if _annotation_code_at(annotations, i) == paradigm_code
    ]

    if not indices_for_code:
        return Interval(start=np.array([]), end=np.array([]))

    paradigm_name = PARADIGM_MAP[paradigm_code][0]
    starts = []
    ends = []
    for i in indices_for_code:
        starts.append(float(annotations.start[i]))
        ends.append(
            _paradigm_segment_end(i, paradigm_code, annotations, paradigm_start_indices)
        )

    event_codes = np.empty(len(starts), dtype=object)
    for j, (s, e) in enumerate(zip(starts, ends)):
        event_codes[j] = _extract_event_codes(annotations, s, e)

    return Interval(
        start=np.array(starts),
        end=np.array(ends),
        description=np.array([paradigm_name] * len(starts), dtype="U"),
        start_code=np.full(len(starts), paradigm_code, dtype=np.int64),
        event_codes=event_codes,
    )


def get_all_paradigm_intervals(annotations: Interval) -> Interval:
    """Map all paradigms in a recording and return their intervals in order.

    Args:
        annotations: Raw annotations from the recording.

    Returns:
        Interval with one segment per paradigm occurrence (start, end,
        description, code, event_codes), in order of appearance.
        ``event_codes`` is a numpy object array where each element is an
        ``int64`` array of all numeric annotation codes within that paradigm.
        Empty Interval if no paradigms.
    """
    if len(annotations) == 0 or annotations.description is None:
        return Interval(start=np.array([]), end=np.array([]))

    n = len(annotations.start)
    paradigm_start_indices = [
        i for i in range(n) if _annotation_code_at(annotations, i) in PARADIGM_MAP
    ]
    if not paradigm_start_indices:
        return Interval(start=np.array([]), end=np.array([]))

    starts = []
    ends = []
    names = []
    codes = []
    for i in paradigm_start_indices:
        code = _annotation_code_at(annotations, i)
        if code is None or code not in PARADIGM_MAP:
            continue
        paradigm_name = PARADIGM_MAP[code][0]
        starts.append(float(annotations.start[i]))
        ends.append(_paradigm_segment_end(i, code, annotations, paradigm_start_indices))
        names.append(paradigm_name)
        codes.append(code)

    event_codes = np.empty(len(starts), dtype=object)
    for j, (s, e) in enumerate(zip(starts, ends)):
        event_codes[j] = _extract_event_codes(annotations, s, e)

    return Interval(
        start=np.array(starts),
        end=np.array(ends),
        description=np.array(names, dtype="U"),
        start_code=np.array(codes, dtype=np.int64),
        event_codes=event_codes,
    )


def get_session_task_from_paradigm_intervals(
    paradigm_intervals: Interval,
) -> Task | None:
    """Return the task for the first paradigm segment, if any.

    Used to set session_description.task when paradigm intervals are available.
    """
    if len(paradigm_intervals) == 0:
        return None
    code = int(paradigm_intervals.code[0])
    if code not in PARADIGM_MAP:
        return None
    return PARADIGM_MAP[code][1]
