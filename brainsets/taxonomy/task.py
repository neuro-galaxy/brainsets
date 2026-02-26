from brainsets.core import StringIntEnum


class Task(StringIntEnum):
    REACHING = 0

    # For datasets where no tasks are involved
    FREE_BEHAVIOR = 1

    # A Shenoy-style task involving handwriting different characters.
    DISCRETE_WRITING_CHARACTER = 2

    DISCRETE_WRITING_LINE = 3

    CONTINUOUS_WRITING = 4

    # Allen data
    DISCRETE_VISUAL_CODING = 5

    # speech
    DISCRETE_SPEAKING_CVSYLLABLE = 6

    # Full sentence speaking
    CONTINUOUS_SPEAKING_SENTENCE = 7

    # EEG tasks
    RESTING_STATE = 8

    SEQUENCE_LEARNING = 9

    VISUAL_SEARCH = 10

    SURROUND_SUPPRESSION = 11

    CONTRAST_CHANGE_DETECTION = 12

    NATURALISTIC_VIEWING = 13


class Stimulus(StringIntEnum):
    """Stimuli can variously act like inputs (for conditioning) or like outputs."""

    TARGET2D = 0
    TARGETON = 1
    GO_CUE = 2
    TARGETACQ = 3

    DRIFTING_GRATINGS = 4
