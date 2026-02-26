from pathlib import Path

import pytest

try:
    import mne_bids

    MNE_BIDS_AVAILABLE = True
    from brainsets.utils.bids_utils import (
        EEG_EXTENSIONS,
        IEEG_EXTENSIONS,
        _fetch_recordings,
        build_bids_path,
        check_recording_files_exist,
        fetch_eeg_recordings,
        fetch_ieeg_recordings,
        parse_bids_fname,
        parse_recording_id,
    )
except ImportError:
    MNE_BIDS_AVAILABLE = False
    EEG_EXTENSIONS = None
    IEEG_EXTENSIONS = None
    _fetch_recordings = None
    build_bids_path = None
    check_recording_files_exist = None
    fetch_eeg_recordings = None
    fetch_ieeg_recordings = None
    parse_bids_fname = None
    parse_recording_id = None

s
@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestParseBidsFname:
    def test_parses_valid_bids_filename(self):
        relative_path = "sub-01_ses-02_task-Sleep_acq-headband_run-03_desc-clean_eeg.edf"
        parsed = parse_bids_fname(relative_path, modality="eeg")
        assert parsed == {
            "subject": "01",
            "session": "02",
            "task": "Sleep",
            "acquisition": "headband",
            "run": "03",
            "description": "clean",
        }

    def test_parses_valid_bids_filename_with_bids_root(self):
        full_path = "ds000001/sub-01/ses-02/eeg/sub-01_ses-02_task-Sleep_acq-headband_run-03_desc-clean_eeg.edf"
        parsed = parse_bids_fname(full_path, modality="eeg")
        assert parsed == {
            "subject": "01",
            "session": "02",
            "task": "Sleep",
            "acquisition": "headband",
            "run": "03",
            "description": "clean",
        }
        
    def test_accepts_path_input(self):
        fname = Path("sub-01_task-rest_ieeg.nwb")
        parsed = parse_bids_fname(fname, modality="ieeg")

        assert parsed["subject"] == "01"
        assert parsed["task"] == "rest"

    def test_returns_none_on_modality_mismatch(self):
        fname = "sub-01_task-rest_ieeg.nwb"
        assert parse_bids_fname(fname, modality="eeg") is None

    def test_returns_none_for_invalid_filename(self):
        assert parse_bids_fname("not-a-bids-name", modality="eeg") is None
        
@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestParseRecordingId:
    def test_parses_valid_recording_id(self):
        recording_id = "sub-01_ses-02_task-rest_acq-ecog_run-03_desc-preproc"
        parsed = parse_recording_id(recording_id)

        assert parsed == {
            "subject": "01",
            "session": "02",
            "task": "rest",
            "acquisition": "ecog",
            "run": "03",
            "description": "preproc",
        }

    def test_raises_when_subject_missing(self):
        with pytest.raises(ValueError, match="Invalid recording_id format"):
            parse_recording_id("task-rest")

    def test_raises_when_task_missing(self):
        with pytest.raises(ValueError, match="missing task entity"):
            parse_recording_id("sub-01_ses-01")


@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
class TestFetchRecordings:
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

    def test_raises_for_invalid_recording_id(self, tmp_path):
        with pytest.raises(ValueError, match="missing task entity"):
            build_bids_path(tmp_path, "sub-01", "eeg")


@pytest.mark.skipif(not MNE_BIDS_AVAILABLE, reason="mne_bids not installed")
def test_ieeg_extensions_include_nwb():
    assert ".nwb" in IEEG_EXTENSIONS
