"""Unit tests for OpenNeuro Pipeline classes."""

import pytest
import numpy as np
import pandas as pd
import mne

from pathlib import Path
from unittest.mock import MagicMock, patch, Mock, PropertyMock
from argparse import Namespace

from temporaldata import Data, Interval

from brainsets.utils.openneuro.pipeline import (
    OpenNeuroPipeline,
    OpenNeuroEEGPipeline,
    OpenNeuroIEEGPipeline,
    OpenNeuroContext,
)
from brainsets.descriptions import (
    BrainsetDescription,
    SubjectDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.taxonomy import Species


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files."""
    return tmp_path


@pytest.fixture
def mock_args_no_reprocessing():
    """Mock args with redownload and reprocess set to False."""
    return Namespace(redownload=False, reprocess=False)


@pytest.fixture
def mock_args_with_reprocessing():
    """Mock args with redownload and reprocess set to True."""
    return Namespace(redownload=True, reprocess=True)


@pytest.fixture
def manifest_row():
    """Mock manifest row with typical structure."""
    row = MagicMock()
    row.Index = "rec-001"
    row.subject_id = "sub-01"
    row.s3_url = "s3://openneuro.org/ds005085/sub-01/eeg/rec-001"
    return row


@pytest.fixture
def participants_df():
    """Mock participants DataFrame."""
    return pd.DataFrame(
        {
            "participant_id": ["sub-01", "sub-02"],
            "age": [25, 30],
            "sex": ["M", "F"],
        }
    ).set_index("participant_id")


@pytest.fixture
def mock_raw():
    """Mock MNE raw object."""
    raw = MagicMock(spec=mne.io.BaseRaw)
    raw.info = {"sfreq": 250.0, "meas_date": None, "bads": ["EEG_BAD"]}
    raw.get_data.return_value = np.random.randn(10, 1000)
    raw.ch_names = [
        "EEG_1",
        "EEG_2",
        "EEG_3",
        "EEG_IGNORE",
        "EEG_BAD",
        "EOG_L",
        "EOG_R",
        "EMG",
        "STIM",
    ] + ["unused"] * 3
    raw.get_channel_types.return_value = ["eeg"] * len(raw.ch_names)
    raw.times = np.linspace(0, 4, 1000)
    raw.get_montage.return_value = None
    return raw


@pytest.fixture
def eeg_pipeline_class():
    """Concrete EEG pipeline class for testing."""

    class TestEEGPipeline(OpenNeuroEEGPipeline):
        dataset_id = "ds005085"
        brainset_id = "test_eeg_brainset"
        origin_version = "1.0.0"

    return TestEEGPipeline


@pytest.fixture
def ieeg_pipeline_class():
    """Concrete iEEG pipeline class for testing."""

    class TestIEEGPipeline(OpenNeuroIEEGPipeline):
        dataset_id = "ds005085"
        brainset_id = "test_ieeg_brainset"
        origin_version = "1.0.0"

    return TestIEEGPipeline


@pytest.fixture
def eeg_pipeline_instance(eeg_pipeline_class, temp_dir, mock_args_no_reprocessing):
    """Instantiated EEG pipeline."""
    instance = eeg_pipeline_class(
        raw_dir=temp_dir / "raw",
        processed_dir=temp_dir / "processed",
        args=mock_args_no_reprocessing,
    )
    return instance


@pytest.fixture
def ieeg_pipeline_instance(ieeg_pipeline_class, temp_dir, mock_args_no_reprocessing):
    """Instantiated iEEG pipeline."""
    instance = ieeg_pipeline_class(
        raw_dir=temp_dir / "raw",
        processed_dir=temp_dir / "processed",
        args=mock_args_no_reprocessing,
    )
    return instance


# ============================================================================
# Tests for OpenNeuroEEGPipeline
# ============================================================================


class TestOpenNeuroEEGPipeline:
    """Tests for OpenNeuroEEGPipeline class."""

    def test_modality_is_eeg(self, eeg_pipeline_class):
        """EEG pipeline has 'eeg' modality."""
        assert eeg_pipeline_class.modality == "eeg"

    def test_required_attributes_present(self, eeg_pipeline_class):
        """Pipeline has required class attributes."""
        assert eeg_pipeline_class.dataset_id == "ds005085"
        assert eeg_pipeline_class.brainset_id == "test_eeg_brainset"
        assert eeg_pipeline_class.origin_version == "1.0.0"


class TestOpenNeuroIEEGPipeline:
    """Tests for OpenNeuroIEEGPipeline class."""

    def test_modality_is_ieeg(self, ieeg_pipeline_class):
        """iEEG pipeline has 'ieeg' modality."""
        assert ieeg_pipeline_class.modality == "ieeg"

    def test_required_attributes_present(self, ieeg_pipeline_class):
        """Pipeline has required class attributes."""
        assert ieeg_pipeline_class.dataset_id == "ds005085"
        assert ieeg_pipeline_class.brainset_id == "test_ieeg_brainset"
        assert ieeg_pipeline_class.origin_version == "1.0.0"


# ============================================================================
# Tests for shared OpenNeuro context caching
# ============================================================================


class TestVersionMismatchPolicyForwarding:
    """Tests for forwarding on_version_mismatch policy through pipeline methods."""

    @patch.object(OpenNeuroPipeline, "validate_dataset_id")
    @patch.object(OpenNeuroPipeline, "_validate_dataset_version")
    @patch("brainsets.utils.openneuro.pipeline.fetch_latest_snapshot_tag")
    @patch("brainsets.utils.openneuro.pipeline.fetch_participants_tsv")
    @patch("brainsets.utils.openneuro.pipeline.fetch_species")
    def test_openneuro_context_forwards_on_version_mismatch_abort(
        self,
        mock_species,
        mock_part,
        mock_fetch_tag,
        mock_ver,
        mock_id,
        eeg_pipeline_class,
    ):
        """_openneuro_context passes on_version_mismatch='abort' to validate_dataset_version."""
        eeg_pipeline_class._cached_openneuro_context.clear()
        mock_id.return_value = "ds005085"
        mock_fetch_tag.return_value = "1.0.0"
        mock_ver.return_value = "1.0.0"
        mock_part.return_value = None
        mock_species.return_value = "homo sapiens"

        eeg_pipeline_class._openneuro_context(on_version_mismatch="abort")

        mock_ver.assert_called_once()
        call_kwargs = mock_ver.call_args[1]
        assert call_kwargs["on_mismatch"] == "abort"

    @patch.object(OpenNeuroPipeline, "validate_dataset_id")
    @patch.object(OpenNeuroPipeline, "_validate_dataset_version")
    @patch("brainsets.utils.openneuro.pipeline.fetch_participants_tsv")
    @patch("brainsets.utils.openneuro.pipeline.fetch_species")
    def test_openneuro_context_defaults_to_prompt(
        self,
        mock_species,
        mock_part,
        mock_ver,
        mock_id,
        eeg_pipeline_class,
    ):
        """_openneuro_context defaults to on_version_mismatch='prompt'."""
        eeg_pipeline_class._cached_openneuro_context.clear()
        mock_id.return_value = "ds005085"
        mock_ver.return_value = "1.0.0"
        mock_part.return_value = None
        mock_species.return_value = "homo sapiens"

        eeg_pipeline_class._openneuro_context()

        mock_ver.assert_called_once()
        call_kwargs = mock_ver.call_args[1]
        assert call_kwargs["on_mismatch"] == "prompt"

    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    @patch.object(OpenNeuroPipeline, "_validate_dataset_version")
    @patch.object(OpenNeuroPipeline, "validate_dataset_id")
    def test_get_manifest_passes_on_version_mismatch_from_args(
        self,
        mock_id,
        mock_ver,
        mock_fetch_eeg,
        mock_fetch_files,
        eeg_pipeline_class,
        temp_dir,
    ):
        """get_manifest passes args.on_version_mismatch to _openneuro_context."""
        from argparse import Namespace

        args = Namespace(on_version_mismatch="continue", redownload=False, reprocess=False)
        mock_id.return_value = "ds005085"
        mock_ver.return_value = "1.0.0"
        mock_fetch_files.return_value = ["sub-01/eeg/rec-001_eeg.edf"]
        mock_fetch_eeg.return_value = []

        eeg_pipeline_class._cached_openneuro_context.clear()
        with patch.object(
            eeg_pipeline_class, "_openneuro_context", return_value={
                "latest_snapshot_tag": "1.0.0",
                "participants_data": None,
                "species": "homo sapiens",
            }
        ) as mock_ctx:
            try:
                eeg_pipeline_class.get_manifest(temp_dir, args)
            except ValueError:
                pass

            mock_ctx.assert_called_once()
            call_kwargs = mock_ctx.call_args[1]
            assert call_kwargs["on_version_mismatch"] == "continue"

    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_defaults_to_prompt_when_args_none(
        self,
        mock_fetch_eeg,
        mock_fetch_files,
        eeg_pipeline_class,
        temp_dir,
    ):
        """get_manifest defaults to 'prompt' when args is None."""
        mock_fetch_files.return_value = ["sub-01/eeg/rec-001_eeg.edf"]
        mock_fetch_eeg.return_value = []

        eeg_pipeline_class._cached_openneuro_context.clear()
        with patch.object(
            eeg_pipeline_class, "_openneuro_context", return_value={
                "latest_snapshot_tag": "1.0.0",
                "participants_data": None,
                "species": "homo sapiens",
            }
        ) as mock_ctx:
            try:
                eeg_pipeline_class.get_manifest(temp_dir, None)
            except ValueError:
                pass

            mock_ctx.assert_called_once()
            call_kwargs = mock_ctx.call_args[1]
            assert call_kwargs["on_version_mismatch"] == "prompt"


class TestValidateDatasetId:
    """Tests for OpenNeuroPipeline.validate_dataset_id helper method."""

    def test_valid_strict_format_passes(self):
        """Valid strict format (ds + 6 digits) does not raise."""
        OpenNeuroPipeline.validate_dataset_id("ds005085")

    def test_another_valid_strict_format_passes(self):
        """Another valid strict format with different digits does not raise."""
        OpenNeuroPipeline.validate_dataset_id("ds000001")

    def test_max_valid_numeric_value_passes(self):
        """Maximum valid numeric value (009999) does not raise."""
        OpenNeuroPipeline.validate_dataset_id("ds009999")

    def test_uppercase_prefix_raises_error(self):
        """Uppercase 'DS' prefix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("DS005085")

    def test_whitespace_around_input_raises_error(self):
        """Whitespace around input raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("  ds005085  ")

    def test_missing_prefix_raises_error(self):
        """Missing 'ds' prefix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("005085")

    def test_non_numeric_suffix_raises_error(self):
        """Non-numeric suffix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("ds00a00")

    def test_too_few_digits_raises_error(self):
        """Too few digits after 'ds' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("ds5085")

    def test_too_many_digits_raises_error(self):
        """Too many digits after 'ds' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("ds0050850")

    def test_numeric_part_exceeds_range_raises_error(self):
        """Numeric part exceeding 9999 raises ValueError."""
        with pytest.raises(ValueError, match="invalid numeric portion"):
            OpenNeuroPipeline.validate_dataset_id("ds010000")

    def test_numeric_part_below_minimum_raises_error(self):
        """Numeric part of 000000 raises ValueError."""
        with pytest.raises(ValueError, match="invalid numeric portion"):
            OpenNeuroPipeline.validate_dataset_id("ds000000")

    def test_invalid_format_no_prefix_raises_error(self):
        """Invalid format without 'ds' prefix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("invalid")

    def test_empty_string_raises_error(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("")

    def test_whitespace_only_raises_error(self):
        """Whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            OpenNeuroPipeline.validate_dataset_id("   ")


class TestValidateDatasetVersion:
    """Tests for OpenNeuroPipeline._validate_dataset_version helper method."""

    class _VersionTestPipeline(OpenNeuroEEGPipeline):
        dataset_id = "ds005085"
        brainset_id = "test_eeg_brainset"
        origin_version = "1.0.0"

    def test_returns_when_versions_match(self):
        """Returns cleanly when latest tag matches origin version."""
        result = self._VersionTestPipeline._validate_dataset_version("1.0.0")
        assert result is None

    def test_warns_when_versions_differ_with_continue(self, caplog):
        """Logs warning when version differs and policy is 'continue'."""
        result = self._VersionTestPipeline._validate_dataset_version(
            "2.0.0", on_mismatch="continue"
        )

        assert result is None
        assert "version '1.0.0' was used to create the brainset pipeline" in caplog.text
        assert "but the latest available version on OpenNeuro is '2.0.0'" in caplog.text

    def test_on_mismatch_abort_raises_error(self):
        """on_mismatch='abort' exits cleanly when versions differ."""
        with pytest.raises(SystemExit, match="Aborting pipeline due to dataset version mismatch"):
            self._VersionTestPipeline._validate_dataset_version(
                "2.0.0", on_mismatch="abort"
            )

    def test_on_mismatch_continue_warns_and_returns(self, caplog):
        """on_mismatch='continue' logs warning and returns latest version."""
        result = self._VersionTestPipeline._validate_dataset_version(
            "2.0.0", on_mismatch="continue"
        )

        assert result is None
        assert "version '1.0.0' was used to create the brainset pipeline" in caplog.text

    @patch("builtins.input", return_value="y")
    @patch("sys.stdin.isatty", return_value=True)
    def test_on_mismatch_prompt_interactive_accept_continues(
        self, mock_isatty, mock_input
    ):
        """on_mismatch='prompt' with interactive TTY and 'y' continues."""
        result = self._VersionTestPipeline._validate_dataset_version(
            "2.0.0", on_mismatch="prompt"
        )

        assert result is None
        mock_input.assert_called_once()
        assert "Continue with latest version?" in mock_input.call_args[0][0]

    @patch("builtins.input", return_value="n")
    @patch("sys.stdin.isatty", return_value=True)
    def test_on_mismatch_prompt_interactive_reject_aborts(
        self, mock_isatty, mock_input
    ):
        """on_mismatch='prompt' with interactive TTY and 'n' aborts."""
        with pytest.raises(SystemExit, match="Aborted by user due to dataset version mismatch"):
            self._VersionTestPipeline._validate_dataset_version(
                "2.0.0", on_mismatch="prompt"
            )

        mock_input.assert_called_once()

    @patch("sys.stdin.isatty", return_value=False)
    def test_on_mismatch_prompt_non_interactive_aborts(self, mock_isatty):
        """on_mismatch='prompt' with non-interactive TTY aborts with clear message."""
        with pytest.raises(
            SystemExit,
            match="non-interactive session",
        ):
            self._VersionTestPipeline._validate_dataset_version(
                "2.0.0", on_mismatch="prompt"
            )

    @patch("builtins.input")
    @patch("sys.stdin.isatty")
    def test_on_mismatch_prompt_no_prompt_when_versions_match(
        self, mock_isatty, mock_input
    ):
        """on_mismatch='prompt' does not prompt or check TTY when versions match."""
        result = self._VersionTestPipeline._validate_dataset_version(
            "1.0.0", on_mismatch="prompt"
        )

        assert result is None
        mock_isatty.assert_not_called()
        mock_input.assert_not_called()

    @patch("builtins.input", return_value="yes")
    @patch("sys.stdin.isatty", return_value=True)
    def test_on_mismatch_prompt_accepts_yes_variant(
        self, mock_isatty, mock_input
    ):
        """on_mismatch='prompt' accepts 'yes' as valid confirmation."""
        result = self._VersionTestPipeline._validate_dataset_version(
            "2.0.0", on_mismatch="prompt"
        )

        assert result is None


class TestNormalizeSpecies:
    """Tests for OpenNeuroPipeline._normalize_species helper method."""

    @pytest.mark.parametrize(
        "species",
        ["homo", "homo sapiens", "human", "humans", "H. sapiens", "  HUMAN  ", "h. sapiens"],
    )
    def test_returns_homo_sapiens_for_supported_aliases(self, species):
        """Supported aliases normalize to canonical homo sapiens."""
        assert OpenNeuroPipeline._normalize_species(species) == "homo sapiens"

    @pytest.mark.parametrize("species", ["mus musculus", "canis lupus", "", None, 42])
    def test_returns_unknown_for_non_human_or_invalid_values(self, species):
        """Non-human or invalid values normalize to unknown."""
        assert OpenNeuroPipeline._normalize_species(species) == "unknown"


class TestOpenNeuroContext:
    """Tests for _openneuro_context caching."""

    def test_shared_context_returns_openneuro_context_type(self, eeg_pipeline_class):
        """_openneuro_context returns OpenNeuroContext dict."""
        with patch.object(OpenNeuroPipeline, "validate_dataset_id") as mock_id:
            with patch.object(OpenNeuroPipeline, "_validate_dataset_version") as mock_ver:
                with patch(
                    "brainsets.utils.openneuro.pipeline.fetch_participants_tsv"
                ) as mock_part:
                    with patch(
                        "brainsets.utils.openneuro.pipeline.fetch_species"
                    ) as mock_species:
                        mock_id.return_value = "ds005085"
                        mock_ver.return_value = "1.0.0"
                        mock_part.return_value = None
                        mock_species.return_value = "homo sapiens"

                        eeg_pipeline_class._cached_openneuro_context.clear()
                        ctx = eeg_pipeline_class._openneuro_context()

        assert isinstance(ctx, dict)
        assert "latest_snapshot_tag" in ctx
        assert "participants_data" in ctx
        assert "species" in ctx

    def test_shared_context_caches_result(self, eeg_pipeline_class):
        """_openneuro_context caches results and doesn't recompute."""
        eeg_pipeline_class._cached_openneuro_context.clear()

        with patch.object(OpenNeuroPipeline, "validate_dataset_id") as mock_id:
            with patch.object(OpenNeuroPipeline, "_validate_dataset_version") as mock_ver:
                with patch(
                    "brainsets.utils.openneuro.pipeline.fetch_participants_tsv"
                ) as mock_part:
                    with patch(
                        "brainsets.utils.openneuro.pipeline.fetch_species"
                    ) as mock_species:
                        mock_id.return_value = "ds005085"
                        mock_ver.return_value = "1.0.0"
                        mock_part.return_value = None
                        mock_species.return_value = "homo sapiens"

                        ctx1 = eeg_pipeline_class._openneuro_context()
                        ctx2 = eeg_pipeline_class._openneuro_context()

        assert ctx1 is ctx2
        # validate_dataset_id is called before cache lookup in each invocation.
        assert mock_id.call_count == 2
        assert mock_ver.call_count == 1
        assert mock_part.call_count == 1
        assert mock_species.call_count == 1

    def test_shared_context_cache_keyed_by_dataset_id(
        self, eeg_pipeline_class, ieeg_pipeline_class
    ):
        """Shared context caches separately per dataset_id."""
        eeg_pipeline_class._cached_openneuro_context.clear()

        with patch.object(OpenNeuroPipeline, "validate_dataset_id") as mock_id:
            with patch.object(OpenNeuroPipeline, "_validate_dataset_version") as mock_ver:
                with patch(
                    "brainsets.utils.openneuro.pipeline.fetch_participants_tsv"
                ) as mock_part:
                    with patch(
                        "brainsets.utils.openneuro.pipeline.fetch_species"
                    ) as mock_species:
                        mock_id.return_value = "ds005085"
                        mock_ver.return_value = "1.0.0"
                        mock_part.return_value = None
                        mock_species.return_value = "homo sapiens"

                        ctx_eeg = eeg_pipeline_class._openneuro_context()

        assert "ds005085" in eeg_pipeline_class._cached_openneuro_context


class TestGetManifest:
    """Tests for get_manifest class method."""

    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_eeg_success(
        self,
        mock_fetch_eeg,
        mock_fetch_files,
        eeg_pipeline_class,
        temp_dir,
    ):
        """get_manifest successfully generates manifest for EEG dataset."""
        mock_fetch_files.return_value = [
            "sub-01/eeg/sub-01_task-rest_eeg.edf",
            "sub-01/eeg/sub-01_task-math_eeg.edf",
            "sub-02/eeg/sub-02_task-rest_eeg.edf",
            "participants.tsv",
            "README",
        ]
        mock_fetch_eeg.return_value = [
            {
                "subject_id": "sub-01",
                "recording_id": "sub-01_task-rest",
                "fpath": "sub-01/eeg/sub-01_task-rest_eeg.edf",
            },
            {
                "subject_id": "sub-01",
                "recording_id": "sub-01_task-math",
                "fpath": "sub-01/eeg/sub-01_task-math_eeg.edf",
            },
            {
                "subject_id": "sub-02",
                "recording_id": "sub-02_task-rest",
                "fpath": "sub-02/eeg/sub-02_task-rest_eeg.edf",
            },
        ]

        eeg_pipeline_class._cached_openneuro_context.clear()
        with patch.object(eeg_pipeline_class, "_openneuro_context") as mock_ctx:
            mock_ctx.return_value = {
                "latest_snapshot_tag": "1.0.0",
                "participants_data": None,
                "species": "homo sapiens",
            }
            with patch(
                "brainsets.utils.openneuro.pipeline.construct_s3_url_from_path"
            ) as mock_s3:

                def s3_url_side_effect(dataset_id, fpath, recording_id):
                    parent_dir = str(Path(fpath).parent)
                    return (
                        f"s3://openneuro.org/{dataset_id}/{parent_dir}/{recording_id}"
                    )

                mock_s3.side_effect = s3_url_side_effect

                result = eeg_pipeline_class.get_manifest(temp_dir, None)

        assert isinstance(result, pd.DataFrame)
        for rec in [
            ("sub-01_task-rest", "sub-01", "sub-01/eeg", "sub-01_task-rest"),
            ("sub-01_task-math", "sub-01", "sub-01/eeg", "sub-01_task-math"),
            ("sub-02_task-rest", "sub-02", "sub-02/eeg", "sub-02_task-rest"),
        ]:
            recording_id, subject_id, sub_dir, rec_id = rec
            assert recording_id in result.index
            assert result.loc[recording_id, "subject_id"] == subject_id
            expected_s3_url = f"s3://openneuro.org/ds005085/{sub_dir}/{rec_id}"
            assert result.loc[recording_id, "s3_url"] == expected_s3_url

    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_ieeg_recordings")
    def test_get_manifest_ieeg_success(
        self,
        mock_fetch_ieeg,
        mock_fetch_files,
        ieeg_pipeline_class,
        temp_dir,
    ):
        """get_manifest successfully generates manifest for iEEG dataset."""
        mock_fetch_files.return_value = [
            "sub-01/ieeg/sub-01_task-rest_ieeg.edf",
            "sub-01/ieeg/sub-01_task-math_ieeg.edf",
            "sub-02/ieeg/sub-02_task-rest_ieeg.edf",
            "participants.tsv",
            "README",
        ]
        mock_fetch_ieeg.return_value = [
            {
                "subject_id": "sub-01",
                "recording_id": "sub-01_task-rest",
                "fpath": "sub-01/ieeg/sub-01_task-rest_ieeg.edf",
            },
            {
                "subject_id": "sub-01",
                "recording_id": "sub-01_task-math",
                "fpath": "sub-01/ieeg/sub-01_task-math_ieeg.edf",
            },
            {
                "subject_id": "sub-02",
                "recording_id": "sub-02_task-rest",
                "fpath": "sub-02/ieeg/sub-02_task-rest_ieeg.edf",
            },
        ]

        ieeg_pipeline_class._cached_openneuro_context.clear()
        with patch.object(ieeg_pipeline_class, "_openneuro_context") as mock_ctx:
            mock_ctx.return_value = {
                "latest_snapshot_tag": "1.0.0",
                "participants_data": None,
                "species": "homo sapiens",
            }
            with patch(
                "brainsets.utils.openneuro.pipeline.construct_s3_url_from_path"
            ) as mock_s3:

                def s3_url_side_effect(dataset_id, fpath, recording_id):
                    parent_dir = str(Path(fpath).parent)
                    return (
                        f"s3://openneuro.org/{dataset_id}/{parent_dir}/{recording_id}"
                    )

                mock_s3.side_effect = s3_url_side_effect

                result = ieeg_pipeline_class.get_manifest(temp_dir, None)

        assert isinstance(result, pd.DataFrame)
        for rec in [
            ("sub-01_task-rest", "sub-01", "sub-01/ieeg", "sub-01_task-rest"),
            ("sub-01_task-math", "sub-01", "sub-01/ieeg", "sub-01_task-math"),
            ("sub-02_task-rest", "sub-02", "sub-02/ieeg", "sub-02_task-rest"),
        ]:
            recording_id, subject_id, sub_dir, rec_id = rec
            assert recording_id in result.index
            assert result.loc[recording_id, "subject_id"] == subject_id
            expected_s3_url = f"s3://openneuro.org/ds005085/{sub_dir}/{rec_id}"
            assert result.loc[recording_id, "s3_url"] == expected_s3_url

    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_with_custom_openneuro_context(
        self,
        mock_fetch_eeg,
        mock_fetch_files,
        temp_dir,
    ):
        """get_manifest works when _openneuro_context is pre-populated."""

        class CustomContextPipeline(OpenNeuroEEGPipeline):
            dataset_id = "ds005085"
            brainset_id = "test_eeg"
            origin_version = "1.0.0"

        mock_fetch_files.return_value = [
            "sub-01/eeg/rec-001_eeg.edf",
            "sub-02/eeg/rec-001_eeg.edf",
        ]
        mock_fetch_eeg.return_value = [
            {
                "subject_id": "sub-01",
                "recording_id": "rec-001",
                "fpath": "sub-01/eeg/rec-001_eeg.edf",
            },
            {
                "subject_id": "sub-02",
                "recording_id": "rec-001",
                "fpath": "sub-02/eeg/rec-001_eeg.edf",
            },
        ]

        CustomContextPipeline._cached_openneuro_context.clear()
        with patch.object(CustomContextPipeline, "_openneuro_context") as mock_ctx:
            mock_ctx.return_value = {
                "latest_snapshot_tag": "1.0.0",
                "participants_data": None,
                "species": "homo sapiens",
            }
            with patch(
                "brainsets.utils.openneuro.pipeline.construct_s3_url_from_path"
            ) as mock_s3:
                mock_s3.return_value = "https://example.com/rec-001"
                result = CustomContextPipeline.get_manifest(temp_dir, None)

        assert len(result) == 2
        assert "rec-001" in result.index

    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    def test_get_manifest_raises_on_unknown_modality(self, mock_fetch_files, temp_dir):
        """get_manifest raises ValueError for unknown modality."""

        class BadPipeline(OpenNeuroPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            modality = "unknown"

        mock_fetch_files.return_value = []

        BadPipeline._cached_openneuro_context.clear()
        with patch.object(BadPipeline, "_openneuro_context") as mock_ctx:
            mock_ctx.return_value = {
                "latest_snapshot_tag": "1.0.0",
                "participants_data": None,
                "species": "homo sapiens",
            }
            with pytest.raises(ValueError, match="Unknown modality"):
                BadPipeline.get_manifest(temp_dir, None)

    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_raises_on_no_recordings_found(
        self,
        mock_fetch_eeg,
        mock_fetch_files,
        eeg_pipeline_class,
        temp_dir,
    ):
        """get_manifest raises ValueError when no recordings are found."""
        mock_fetch_files.return_value = []
        mock_fetch_eeg.return_value = []

        eeg_pipeline_class._cached_openneuro_context.clear()
        with patch.object(eeg_pipeline_class, "_openneuro_context") as mock_ctx:
            mock_ctx.return_value = {
                "latest_snapshot_tag": "1.0.0",
                "participants_data": None,
                "species": "homo sapiens",
            }
            with pytest.raises(ValueError, match="No EEG recordings found"):
                eeg_pipeline_class.get_manifest(temp_dir, None)

    @patch("brainsets.utils.openneuro.pipeline.fetch_all_filenames")
    @patch("brainsets.utils.openneuro.pipeline.fetch_eeg_recordings")
    def test_get_manifest_raises_on_no_recordings_returned_by_fetch(
        self,
        mock_fetch_eeg,
        mock_fetch_files,
        temp_dir,
    ):
        """get_manifest raises ValueError when recording parser returns no rows."""

        class EmptyRecordingsPipeline(OpenNeuroEEGPipeline):
            dataset_id = "ds005085"
            brainset_id = "test_eeg"
            origin_version = "1.0.0"

        mock_fetch_files.return_value = ["sub-01/eeg/rec-001.edf"]
        mock_fetch_eeg.return_value = []

        EmptyRecordingsPipeline._cached_openneuro_context.clear()
        with patch.object(EmptyRecordingsPipeline, "_openneuro_context") as mock_ctx:
            mock_ctx.return_value = {
                "latest_snapshot_tag": "1.0.0",
                "participants_data": None,
                "species": "homo sapiens",
            }
            with pytest.raises(ValueError, match="No EEG recordings found"):
                EmptyRecordingsPipeline.get_manifest(temp_dir, None)


# ============================================================================
# Tests for download
# ============================================================================


class TestDownload:
    """Tests for download method."""

    def test_download_creates_raw_dir(self, eeg_pipeline_instance, manifest_row):
        """download creates raw_dir if it doesn't exist."""
        with patch("brainsets.utils.openneuro.pipeline.download_recording"):
            with patch(
                "brainsets.utils.openneuro.pipeline.download_dataset_description"
            ):
                with patch(
                    "brainsets.utils.openneuro.pipeline.check_eeg_recording_files_exist",
                    return_value=False,
                ):
                    eeg_pipeline_instance.download(manifest_row)

        assert eeg_pipeline_instance.raw_dir.exists()

    def test_download_returns_dict_with_required_keys(
        self, eeg_pipeline_instance, manifest_row
    ):
        """download returns dict with subject_id and recording_id."""
        with patch("brainsets.utils.openneuro.pipeline.download_recording"):
            with patch(
                "brainsets.utils.openneuro.pipeline.download_dataset_description"
            ):
                with patch(
                    "brainsets.utils.openneuro.pipeline.check_eeg_recording_files_exist",
                    return_value=False,
                ):
                    result = eeg_pipeline_instance.download(manifest_row)

        assert "subject_id" in result
        assert "recording_id" in result
        assert result["subject_id"] == "sub-01"
        assert result["recording_id"] == "rec-001"

    def test_download_eeg_skips_if_files_exist_and_no_redownload(
        self, temp_dir, mock_args_no_reprocessing, eeg_pipeline_class, manifest_row
    ):
        """download skips if files exist and redownload is False."""
        pipeline = eeg_pipeline_class(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=mock_args_no_reprocessing,
        )

        with patch(
            "brainsets.utils.openneuro.pipeline.check_eeg_recording_files_exist",
            return_value=True,
        ):
            with patch(
                "brainsets.utils.openneuro.pipeline.download_recording"
            ) as mock_download:
                result = pipeline.download(manifest_row)

        mock_download.assert_not_called()
        assert result["recording_id"] == "rec-001"

    def test_download_ieeg_skips_if_files_exist_and_no_redownload(
        self, temp_dir, mock_args_no_reprocessing, ieeg_pipeline_class, manifest_row
    ):
        """download skips for iEEG if files exist and redownload is False."""
        pipeline = ieeg_pipeline_class(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=mock_args_no_reprocessing,
        )

        with patch(
            "brainsets.utils.openneuro.pipeline.check_ieeg_recording_files_exist",
            return_value=True,
        ):
            with patch(
                "brainsets.utils.openneuro.pipeline.download_recording"
            ) as mock_download:
                result = pipeline.download(manifest_row)

        mock_download.assert_not_called()
        assert result["recording_id"] == "rec-001"

    def test_download_eeg_redownloads_when_redownload_true(
        self, temp_dir, mock_args_with_reprocessing, eeg_pipeline_class, manifest_row
    ):
        """download redownloads when redownload=True even if files exist."""
        pipeline = eeg_pipeline_class(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=mock_args_with_reprocessing,
        )

        with patch(
            "brainsets.utils.openneuro.pipeline.download_recording"
        ) as mock_download:
            with patch(
                "brainsets.utils.openneuro.pipeline.download_dataset_description"
            ):
                result = pipeline.download(manifest_row)

        mock_download.assert_called_once()

    def test_download_raises_on_s3_error(self, eeg_pipeline_instance, manifest_row):
        """download raises RuntimeError on download failure."""
        with patch(
            "brainsets.utils.openneuro.pipeline.download_recording",
            side_effect=Exception("S3 error"),
        ):
            with patch(
                "brainsets.utils.openneuro.pipeline.check_eeg_recording_files_exist",
                return_value=False,
            ):
                with pytest.raises(RuntimeError, match="Failed to download"):
                    eeg_pipeline_instance.download(manifest_row)


# ============================================================================
# Tests for channel remapping methods
# ============================================================================


class TestChannelRemapping:
    """Tests for channel remapping methods."""

    def test_get_channel_name_remapping_returns_class_attribute(
        self, eeg_pipeline_instance
    ):
        """get_channel_name_remapping returns CHANNEL_NAME_REMAPPING when defined."""

        class CustomEEGPipeline(OpenNeuroEEGPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            CHANNEL_NAME_REMAPPING = {"PSG_F3": "F3", "PSG_F4": "F4"}

        pipeline = CustomEEGPipeline(
            raw_dir=Path("/tmp/raw"),
            processed_dir=Path("/tmp/processed"),
            args=Namespace(redownload=False, reprocess=False),
        )
        assert pipeline.get_channel_name_remapping() == {"PSG_F3": "F3", "PSG_F4": "F4"}

    def test_get_channel_name_remapping_returns_none_by_default(
        self, eeg_pipeline_instance
    ):
        """get_channel_name_remapping returns None when not defined."""
        assert eeg_pipeline_instance.get_channel_name_remapping() is None

    def test_get_channel_name_remapping_accepts_recording_id(
        self, eeg_pipeline_instance
    ):
        """get_channel_name_remapping accepts recording_id parameter."""
        result = eeg_pipeline_instance.get_channel_name_remapping(
            recording_id="rec-001"
        )
        assert result is None

    def test_get_type_channels_remapping_returns_class_attribute(self):
        """get_type_channels_remapping returns TYPE_CHANNELS_REMAPPING when defined."""

        class CustomEEGPipeline(OpenNeuroEEGPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            TYPE_CHANNELS_REMAPPING = {"EEG": ["F3", "F4"], "EOG": ["EOG"]}

        pipeline = CustomEEGPipeline(
            raw_dir=Path("/tmp/raw"),
            processed_dir=Path("/tmp/processed"),
            args=Namespace(redownload=False, reprocess=False),
        )
        assert pipeline.get_type_channels_remapping() == {
            "EEG": ["F3", "F4"],
            "EOG": ["EOG"],
        }

    def test_get_type_channels_remapping_returns_none_by_default(
        self, eeg_pipeline_instance
    ):
        """get_type_channels_remapping returns None when not defined."""
        assert eeg_pipeline_instance.get_type_channels_remapping() is None

    def test_get_type_channels_remapping_accepts_recording_id(
        self, eeg_pipeline_instance
    ):
        """get_type_channels_remapping accepts recording_id parameter."""
        result = eeg_pipeline_instance.get_type_channels_remapping(
            recording_id="rec-001"
        )
        assert result is None


# ============================================================================
# Tests for generate_splits
# ============================================================================


class TestGenerateSplits:
    """Tests for generate_splits method."""

    def test_generate_splits_creates_train_and_valid_intervals(
        self, eeg_pipeline_instance
    ):
        """generate_splits creates train and valid intervals."""
        starts = np.array([0.0])
        ends = np.array([100.0])
        domain = Interval(start=starts, end=ends)

        result = eeg_pipeline_instance.generate_splits(
            domain=domain,
            subject_id="sub-01",
            session_id="ses-01",
        )

        assert isinstance(result, Data)
        assert hasattr(result, "train")
        assert hasattr(result, "valid")

    def test_generate_splits_respects_split_ratios(self, temp_dir, eeg_pipeline_class):
        """generate_splits respects custom split_ratios."""

        class CustomRatioPipeline(OpenNeuroEEGPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            split_ratios = (0.8, 0.2)

        pipeline = CustomRatioPipeline(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=Namespace(redownload=False, reprocess=False),
        )

        starts = np.array([0.0])
        ends = np.array([100.0])
        domain = Interval(start=starts, end=ends)

        result = pipeline.generate_splits(domain, "sub-01", "ses-01")

        train_duration = result.train.end[0] - result.train.start[0]
        assert np.isclose(train_duration, 80.0)

    def test_generate_splits_raises_on_invalid_split_ratios_length(
        self, eeg_pipeline_instance
    ):
        """generate_splits raises ValueError if split_ratios length != 2."""
        eeg_pipeline_instance.split_ratios = (0.5, 0.3, 0.2)
        starts = np.array([0.0])
        ends = np.array([100.0])
        domain = Interval(start=starts, end=ends)

        with pytest.raises(ValueError, match="must contain exactly two values"):
            eeg_pipeline_instance.generate_splits(domain, "sub-01", "ses-01")

    def test_generate_splits_raises_on_negative_split_ratios(
        self, eeg_pipeline_instance
    ):
        """generate_splits raises ValueError if any split ratio is negative."""
        eeg_pipeline_instance.split_ratios = (0.5, -0.5)
        starts = np.array([0.0])
        ends = np.array([100.0])
        domain = Interval(start=starts, end=ends)

        with pytest.raises(ValueError, match="cannot contain negative values"):
            eeg_pipeline_instance.generate_splits(domain, "sub-01", "ses-01")

    def test_generate_splits_raises_on_non_unit_sum(self, eeg_pipeline_instance):
        """generate_splits raises ValueError if split ratios don't sum to 1.0."""
        eeg_pipeline_instance.split_ratios = (0.5, 0.3)
        starts = np.array([0.0])
        ends = np.array([100.0])
        domain = Interval(start=starts, end=ends)

        with pytest.raises(ValueError, match="must sum to 1.0"):
            eeg_pipeline_instance.generate_splits(domain, "sub-01", "ses-01")

    def test_generate_splits_creates_assignment_splits(self, eeg_pipeline_instance):
        """generate_splits creates intersubject and intersession assignments."""
        starts = np.array([0.0])
        ends = np.array([100.0])
        domain = Interval(start=starts, end=ends)

        result = eeg_pipeline_instance.generate_splits(domain, "sub-01", "ses-01")

        assert hasattr(result, "intersubject_assignment")
        assert hasattr(result, "intersession_assignment")


# ============================================================================
# Tests for _process_common
# ============================================================================


class TestProcessCommon:
    """Tests for _process_common method."""

    @patch("brainsets.utils.openneuro.pipeline.MNE_BIDS_AVAILABLE", False)
    def test_process_common_raises_when_mne_bids_unavailable(
        self, eeg_pipeline_instance
    ):
        """_process_common raises ImportError if mne-bids not available."""
        download_output = {"recording_id": "rec-001", "subject_id": "sub-01"}

        with pytest.raises(ImportError, match="mne-bids"):
            eeg_pipeline_instance._process_common(download_output)

    @patch("brainsets.utils.openneuro.pipeline.read_raw_bids")
    def test_process_common_skips_if_already_processed(
        self, mock_read_raw, eeg_pipeline_instance
    ):
        """_process_common returns None if file already exists and not reprocessing."""
        eeg_pipeline_instance.processed_dir.mkdir(exist_ok=True, parents=True)
        (eeg_pipeline_instance.processed_dir / "rec-001.h5").touch()
        download_output = {"recording_id": "rec-001", "subject_id": "sub-01"}

        result = eeg_pipeline_instance._process_common(download_output)

        assert result is None
        mock_read_raw.assert_not_called()

    @patch("brainsets.utils.openneuro.pipeline.build_bids_path")
    @patch("brainsets.utils.openneuro.pipeline.read_raw_bids")
    @patch("brainsets.utils.openneuro.pipeline.extract_signal")
    @patch("brainsets.utils.openneuro.pipeline.extract_channels")
    @patch("brainsets.utils.openneuro.pipeline.extract_measurement_date")
    @patch("brainsets.utils.openneuro.pipeline.get_subject_info")
    def test_process_common_creates_data_object(
        self,
        mock_subject_info,
        mock_meas_date,
        mock_extract_channels,
        mock_extract_signal,
        mock_read_raw,
        mock_bids_path,
        eeg_pipeline_instance,
        mock_raw,
    ):
        """_process_common creates and returns Data object."""
        eeg_pipeline_instance.processed_dir.mkdir(exist_ok=True, parents=True)
        download_output = {"recording_id": "rec-001", "subject_id": "sub-01"}

        mock_bids_path.return_value = "path/to/bids"
        mock_read_raw.return_value = mock_raw
        mock_subject_info.return_value = {"age": 25, "sex": "M"}
        mock_meas_date.return_value = "2023-01-01"
        mock_extract_signal.return_value = MagicMock(
            domain=Interval(start=np.array([0.0]), end=np.array([100.0]))
        )
        mock_extract_channels.return_value = {}

        ctx = {
            "latest_snapshot_tag": "1.0.0",
            "participants_data": None,
            "species": "homo sapiens",
        }

        type(eeg_pipeline_instance)._cached_openneuro_context = {
            eeg_pipeline_instance.dataset_id: ctx
        }
        with patch.object(
            eeg_pipeline_instance, "generate_splits", return_value=MagicMock()
        ):
            result = eeg_pipeline_instance._process_common(download_output)

        assert result is not None
        assert len(result) == 2
        data, store_path = result
        assert isinstance(data, Data)
        assert isinstance(store_path, Path)

    @patch("brainsets.utils.openneuro.pipeline.build_bids_path")
    @patch("brainsets.utils.openneuro.pipeline.read_raw_bids")
    @patch("brainsets.utils.openneuro.pipeline.get_subject_info")
    @patch("brainsets.utils.openneuro.pipeline.extract_measurement_date")
    @patch("brainsets.utils.openneuro.pipeline.extract_signal")
    def test_channel_and_type_remapping_and_ignore_channels(
        self,
        mock_extract_signal,
        mock_meas_date,
        mock_subject_info,
        mock_read_raw,
        mock_bids_path,
        temp_dir,
        mock_args_no_reprocessing,
        mock_raw,
    ):
        """Test that _process_common remaps channel names/types and applies ignore channels."""

        class CustomEEGPipeline(OpenNeuroEEGPipeline):
            dataset_id = "ds005085"
            brainset_id = "test"
            origin_version = "1.0.0"
            CHANNEL_NAME_REMAPPING = {"EEG_1": "F3", "EEG_4": "F4"}
            TYPE_CHANNELS_REMAPPING = {
                "EEG": ["F3", "EEG_2", "EEG_3", "F4"],
                "EOG": ["EOG_L", "EOG_R", "EOG"],
                "EMG": ["EMG"],
                "STIM": ["STIM"],
                "MISC": ["unused"],
            }
            IGNORE_CHANNELS = ["EEG_IGNORE"]

        pipeline = CustomEEGPipeline(
            raw_dir=temp_dir / "raw",
            processed_dir=temp_dir / "processed",
            args=mock_args_no_reprocessing,
        )

        download_output = {"recording_id": "rec-001", "subject_id": "sub-01"}

        mock_bids_path.return_value = "path/to/bids_path"
        mock_read_raw.return_value = mock_raw
        mock_subject_info.return_value = {"age": 25, "sex": "M"}
        mock_meas_date.return_value = "2023-01-01"
        mock_extract_signal.return_value = MagicMock(
            domain=Interval(start=np.array([0.0]), end=np.array([100.0]))
        )

        ctx = {
            "latest_snapshot_tag": "1.0.0",
            "participants_data": None,
            "species": "homo sapiens",
        }

        type(pipeline)._cached_openneuro_context = {pipeline.dataset_id: ctx}
        with patch.object(
            pipeline, "generate_splits", return_value=MagicMock()
        ):
            data, path = pipeline._process_common(download_output)

        assert all(
            data.channels.id
            == np.array(
                ["F3", "EEG_2", "EEG_3", "EEG_BAD", "EOG_L", "EOG_R", "EMG", "STIM"]
                + ["unused"] * 3
            )
        )
        assert all(
            data.channels.type
            == np.array(
                [
                    "eeg",
                    "eeg",
                    "eeg",
                    "eeg",
                    "eog",
                    "eog",
                    "emg",
                    "stim",
                    "misc",
                    "misc",
                    "misc",
                ]
            )
        )
        assert all(
            data.channels.bad
            == np.array(
                [
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ]
            )
        )


# ============================================================================
# Tests for process
# ============================================================================


class TestProcess:
    """Tests for process method."""

    def test_process_skips_when_process_common_returns_none(
        self, eeg_pipeline_instance
    ):
        """process returns early if _process_common returns None."""
        download_output = {"recording_id": "rec-001", "subject_id": "sub-01"}

        with patch.object(eeg_pipeline_instance, "_process_common", return_value=None):
            with patch("builtins.open") as mock_file:
                eeg_pipeline_instance.process(download_output)

        mock_file.assert_not_called()

    def test_process_saves_data_to_h5(self, eeg_pipeline_instance):
        """process saves Data object to HDF5 file."""
        download_output = {"recording_id": "rec-001", "subject_id": "sub-01"}
        mock_data = MagicMock(spec=Data)
        store_path = eeg_pipeline_instance.processed_dir / "rec-001.h5"

        with patch.object(
            eeg_pipeline_instance,
            "_process_common",
            return_value=(mock_data, store_path),
        ):
            with patch("brainsets.utils.openneuro.pipeline.h5py.File"):
                eeg_pipeline_instance.process(download_output)

        mock_data.to_hdf5.assert_called_once()
