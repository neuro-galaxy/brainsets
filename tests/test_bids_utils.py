from pathlib import Path
from io import StringIO

import pandas as pd
import pytest

try:
    import mne_bids

    MNE_BIDS_AVAILABLE = True
    from brainsets.utils.bids_utils import (
        EEG_EXTENSIONS,
        IEEG_EXTENSIONS,
        fetch_eeg_recordings,
        fetch_ieeg_recordings,
        check_recording_files_exist,
        build_bids_path,
        get_subject_info,
        load_participants_tsv,
        _fetch_recordings,
        _parse_bids_fname,
    )
except ImportError:
    MNE_BIDS_AVAILABLE = False
    EEG_EXTENSIONS = None
    IEEG_EXTENSIONS = None
    fetch_eeg_recordings = None
    fetch_ieeg_recordings = None
    check_recording_files_exist = None
    build_bids_path = None
    get_subject_info = None
    load_participants_tsv = None
    _fetch_recordings = None
    _parse_bids_fname = None


@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestFetchRecordings:
    def test_str_path_candidate_files(self):
        candidate_files = [
            Path("sub-01/eeg/sub-01_task-Sleep_eeg.edf"),
            "sub-01/eeg/sub-01_task-Sleep_channels.tsv",
            "sub-02/eeg/sub-02_ses-01_task-Rest_acq-headband_run-01_eeg.vhdr",
            "sub-02/eeg/sub-02_ses-01_task-Rest_acq-headband_run-01_eeg.vmrk",
            "participants.tsv",
        ]
        recordings = _fetch_recordings(
            EEG_EXTENSIONS,
            "eeg",
            candidate_files=candidate_files,
        )
        rec1 = next(r for r in recordings if r["subject_id"] == "sub-01")
        assert rec1["recording_id"] == "sub-01_task-Sleep"
        assert rec1["task_id"] == "Sleep"
        assert rec1["session_id"] is None
        
        rec2 = next(r for r in recordings if r["subject_id"] == "sub-02")
        assert rec2["recording_id"] == "sub-02_ses-01_task-Rest_acq-headband_run-01"
        assert rec2["task_id"] == "Rest"
        assert rec2["session_id"] == "ses-01"
        assert rec2["acquisition_id"] == "headband"
        assert rec2["run_id"] == "01"
        
    def test_raises_when_bids_root_and_candidate_files_both_provided(self, tmp_path):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _fetch_recordings(
                EEG_EXTENSIONS,
                "eeg",
                bids_root=tmp_path,
                candidate_files=[Path("sub-01_task-rest_eeg.edf")],
            )

    def test_raises_when_bids_root_and_candidate_files_both_missing(self):
        with pytest.raises(ValueError, match="'bids_root' is required"):
            _fetch_recordings(EEG_EXTENSIONS, "eeg")

    def test_filters_by_extension_and_modality_and_deduplicates(self):
        candidate_files = [
            Path("sub-01_task-rest_eeg.edf"),
            Path("sub-01_task-rest_eeg.bdf"),  # duplicate recording_id, supported ext
            Path("sub-01_task-rest_ieeg.nwb"),  # wrong modality
            Path("sub-01_task-rest_events.tsv"),  # unsupported ext
        ]

        recordings = _fetch_recordings(
            EEG_EXTENSIONS,
            "eeg",
            candidate_files=candidate_files,
        )

        assert len(recordings) == 1
        assert recordings[0]["recording_id"] == "sub-01_task-rest"
        assert recordings[0]["subject_id"] == "sub-01"
        assert recordings[0]["session_id"] is None
        assert recordings[0]["task_id"] == "rest"
        assert recordings[0]["eeg_file"] == Path("sub-01_task-rest_eeg.edf")

    def test_fetch_eeg_recordings_wrapper(self):
        recordings = fetch_eeg_recordings(
            candidate_files=[Path("sub-02_task-nback_run-01_eeg.edf")]
        )

        assert len(recordings) == 1
        assert recordings[0]["recording_id"] == "sub-02_task-nback_run-01"
        assert "eeg_file" in recordings[0]

    def test_fetch_ieeg_recordings_wrapper(self):
        recordings = fetch_ieeg_recordings(
            candidate_files=[Path("sub-03_task-VisualNaming_ieeg.nwb")]
        )

        assert len(recordings) == 1
        assert recordings[0]["recording_id"] == "sub-03_task-VisualNaming"
        assert "ieeg_file" in recordings[0]

    def test_no_eeg_files(self):
        candidate_files = [
            Path("participants.tsv"),
            Path("dataset_description.json"),
        ]

        recordings = fetch_eeg_recordings(candidate_files=candidate_files)
        assert len(recordings) == 0

@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestCheckRecordingFilesExist:
    def test_returns_false_when_subject_dir_missing(self, tmp_path):
        missing_dir = tmp_path / "sub-01"
        assert check_recording_files_exist("sub-01_task-rest", missing_dir) is False

    def test_returns_true_when_matching_supported_file_exists(self, tmp_path):
        subject_dir = tmp_path / "sub-01"
        subject_dir.mkdir()
        (subject_dir / "sub-01_task-rest_eeg.EDF").write_text("dummy")

        assert check_recording_files_exist("sub-01_task-rest", subject_dir) is True

    def test_returns_false_when_only_unsupported_files_exist(self, tmp_path):
        subject_dir = tmp_path / "sub-01"
        subject_dir.mkdir()
        (subject_dir / "sub-01_task-rest_events.tsv").write_text("dummy")

        assert check_recording_files_exist("sub-01_task-rest", subject_dir) is False

@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestBuildBidsPath:
    def test_builds_bids_path_from_recording_id(self, tmp_path):
        bids_path = build_bids_path(
            bids_root=tmp_path,
            recording_id="sub-01_ses-02_task-rest_acq-ecog_run-03_desc-preproc",
            modality="ieeg",
        )

        assert bids_path.root == tmp_path
        assert bids_path.subject == "01"
        assert bids_path.session == "02"
        assert bids_path.task == "rest"
        assert bids_path.acquisition == "ecog"
        assert bids_path.run == "03"
        assert bids_path.description == "preproc"
        assert bids_path.datatype == "ieeg"
        assert bids_path.suffix == "ieeg"

    def test_raises_for_missing_task_entity(self, tmp_path):
        with pytest.raises(ValueError, match="missing task entity"):
            build_bids_path(tmp_path, "sub-01_ses-02_acq-ecog_run-03_desc-preproc", "eeg")
    
    def test_raises_for_missing_subject_entity(self, tmp_path):
        with pytest.raises(ValueError, match="missing subject entity"):
            build_bids_path(tmp_path, "task-rest_acq-ecog_run-03_desc-preproc", "eeg")

@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestParseBidsFname:
    def test_parses_valid_bids_filename(self):
        relative_path = "sub-01_ses-02_task-Sleep_acq-headband_run-03_desc-clean_eeg.edf"
        parsed = _parse_bids_fname(relative_path, modality="eeg")
        assert parsed == {
            "subject": "01",
            "session": "02",
            "task": "Sleep",
            "acquisition": "headband",
            "run": "03",
            "description": "clean",
        }

    def test_parses_valid_bids_filename_with_full_path(self):
        full_path = "ds000001/sub-01/ses-02/eeg/sub-01_ses-02_task-Sleep_acq-headband_run-03_desc-clean_eeg.edf"
        parsed = _parse_bids_fname(full_path, modality="eeg")
        assert parsed == {
            "subject": "01",
            "session": "02",
            "task": "Sleep",
            "acquisition": "headband",
            "run": "03",
            "description": "clean",
        }

    def test_pattern_with_missing_entities(self):
        result = _parse_bids_fname("sub-01_task-Sleep_eeg.edf", "eeg")
        assert result is not None
        assert result["subject"] == "01"
        assert result["session"] is None
        assert result["task"] == "Sleep"
        assert result["acquisition"] is None
        assert result["run"] is None
        assert result["description"] is None

    def test_accepts_path_input(self):
        fname = Path("sub-01_task-rest_ieeg.nwb")
        parsed = _parse_bids_fname(fname, modality="ieeg")

        assert parsed["subject"] == "01"
        assert parsed["task"] == "rest"

    def test_returns_none_on_modality_mismatch(self):
        fname = "sub-01_task-rest_ieeg.nwb"
        assert _parse_bids_fname(fname, modality="eeg") is None

    def test_returns_none_for_invalid_filename(self):
        assert _parse_bids_fname("not-a-bids-name", modality="eeg") is None

@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestLoadParticipantsTsv:
    def test_returns_none_when_participants_tsv_missing(self, tmp_path):
        participants_data = load_participants_tsv(tmp_path)
        assert participants_data is None

    def test_returns_none_without_participant_id_column(self, tmp_path):
        participants_tsv = tmp_path / "participants.tsv"
        participants_tsv.write_text("subject_id\tage\tsex\nsub-01\t34\tF\n")

        participants_data = load_participants_tsv(tmp_path)
        assert participants_data is None

    def test_loads_and_indexes_participants_data(self, tmp_path):
        participants_tsv = tmp_path / "participants.tsv"
        participants_tsv.write_text(
            "participant_id\tage\tsex\n"
            "sub-01\t34\tF\n"
            "sub-02\tn/a\tN/A\n"
        )

        participants_data = load_participants_tsv(tmp_path)

        assert participants_data is not None
        assert participants_data.index.name == "participant_id"
        assert list(participants_data.index) == ["sub-01", "sub-02"]
        assert participants_data.loc["sub-01", "age"] == 34
        assert participants_data.loc["sub-01", "sex"] == "F"
        assert pd.isna(participants_data.loc["sub-02", "age"])
        assert pd.isna(participants_data.loc["sub-02", "sex"])

def _participants_df(tsv_content: str) -> pd.DataFrame:
    """Build a participants-like DataFrame with BIDS NA handling."""
    return pd.read_csv(
        StringIO(tsv_content),
        sep="\t",
        na_values=["n/a", "N/A"],
        keep_default_na=True,
    ).set_index("participant_id")

@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestGetSubjectInfo:
    def test_raises_when_bids_root_missing_and_participants_not_provided(self):
        with pytest.raises(ValueError, match="'bids_root' is required"):
            get_subject_info("sub-01")

    def test_returns_none_values_when_participants_tsv_missing(self, tmp_path):
        subject_info = get_subject_info("sub-01", bids_root=tmp_path)
        assert subject_info == {"age": None, "sex": None}

    def test_returns_none_values_when_subject_not_found(self):
        participants_data = _participants_df(
            "participant_id\tage\tsex\nsub-01\t34\tF\n"
        )

        subject_info = get_subject_info("sub-99", participants_data=participants_data)
        assert subject_info == {"age": None, "sex": None}

    def test_returns_subject_age_and_sex_when_available(self):
        participants_data = _participants_df(
            "participant_id\tage\tsex\nsub-01\t34\tF\n"
        )

        subject_info = get_subject_info("sub-01", participants_data=participants_data)
        assert subject_info == {"age": 34, "sex": "F"}

    def test_normalizes_na_values_to_none(self):
        participants_data = _participants_df(
            "participant_id\tage\tsex\nsub-01\tn/a\tN/A\n"
        )

        subject_info = get_subject_info("sub-01", participants_data=participants_data)
        assert subject_info == {"age": None, "sex": None}

    def test_returns_none_for_missing_age_and_sex_columns(self):
        participants_data = _participants_df("participant_id\thandedness\nsub-01\tright\n")

        subject_info = get_subject_info("sub-01", participants_data=participants_data)
        assert subject_info == {"age": None, "sex": None}

@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
def test_ieeg_extensions_include_nwb():
    assert ".nwb" in IEEG_EXTENSIONS
