from brainsets.taxonomy import Task

PARADIGM_MAP = {
    # annotation_code: (paradigm_name, task)
    90: ("Resting Paradigm", Task.RESTING_STATE),
    91: ("Sequence Learning Paradigm", Task.SEQUENCE_LEARNING),
    92: ("Symbol Search Paradigm", Task.VISUAL_SEARCH),
    93: ("Surround Suppression Paradigm Block 1", Task.SURROUND_SUPPRESSION),
    94: ("Contrast Change Paradigm Block 1", Task.CONTRAST_CHANGE_DETECTION),
    95: ("Contrast Change Paradigm Block 2", Task.CONTRAST_CHANGE_DETECTION),
    96: ("Contrast Change Paradigm Block 3", Task.CONTRAST_CHANGE_DETECTION),
    97: ("Surround Suppression Paradigm Block 2", Task.SURROUND_SUPPRESSION),
    81: ("Naturalistic Viewing Paradigm Video 1", Task.NATURALISTIC_VIEWING),
    82: ("Naturalistic Viewing Paradigm Video 2", Task.NATURALISTIC_VIEWING),
    83: ("Naturalistic Viewing Paradigm Video 3", Task.NATURALISTIC_VIEWING),
    84: ("Naturalistic Viewing Paradigm Video 4", Task.NATURALISTIC_VIEWING),
    85: ("Naturalistic Viewing Paradigm Video 5", Task.NATURALISTIC_VIEWING),
    86: ("Naturalistic Viewing Paradigm Video 6", Task.NATURALISTIC_VIEWING),
}

# Annotation codes that mark the end of a paradigm (by paradigm start code).
# Used only for NATURALISTIC_VIEWING paradigms (81-86). Other paradigms end at the
# last annotation before the next paradigm start (or last annotation in recording).
PARADIGM_END_CODES: dict[int, list[int]] = {
    81: [101, 0],
    82: [102, 0],
    83: [103, 0],
    84: [104, 0],
    85: [105, 0],
    86: [106, 0],
}

SAMPLES_COLUMNS = {
    "L Dia X [px]": "l_dia_x",
    "L Dia Y [px]": "l_dia_y",
    "R Dia X [px]": "r_dia_x",
    "R Dia Y [px]": "r_dia_y",
    "L POR X [px]": "l_por_x",
    "L POR Y [px]": "l_por_y",
    "R POR X [px]": "r_por_x",
    "R POR Y [px]": "r_por_y",
    "L Validity": "l_validity",
    "R Validity": "r_validity",
    "Pupil Confidence": "pupil_confidence",
}
