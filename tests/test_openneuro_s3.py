"""Unit tests for OpenNeuro S3 utility functions."""

import pytest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pandas as pd

from brainsets.utils.openneuro.openneuro_s3 import BOTO_AVAILABLE

pytestmark = pytest.mark.skipif(
    not BOTO_AVAILABLE, reason="boto3/botocore not installed"
)

from brainsets.utils.openneuro.openneuro_s3 import (
    ClientError,
    validate_dataset_id,
    validate_dataset_version,
    fetch_all_filenames,
    fetch_participants_tsv,
    construct_s3_url_from_path,
    download_recording,
    download_dataset_description,
    _graphql_query_openneuro,
    OPENNEURO_S3_BUCKET,
)


# ============================================================================
# Shared Fixtures and Helpers
# ============================================================================


@pytest.fixture
def mock_s3_client():
    """Reusable mock S3 client for all S3-related tests."""
    return MagicMock()


def make_client_error(code: str, message: str = "") -> ClientError:
    """Helper to create consistent ClientError instances."""
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "GetObject",
    )


@pytest.fixture
def participants_tsv_bytes():
    """Helper fixture to generate valid participants.tsv content."""

    def _make_tsv(has_participant_id: bool = True) -> bytes:
        if has_participant_id:
            content = (
                "participant_id\tage\tsex\nparticipant_01\t25\tM\nparticipant_02\t30\tF"
            )
        else:
            content = "age\tsex\n25\tM\n30\tF"
        return content.encode("utf-8")

    return _make_tsv


def graphql_ok_response(tag: str) -> dict:
    """Helper to generate valid GraphQL response structure."""
    return {
        "data": {
            "dataset": {
                "latestSnapshot": {
                    "tag": tag,
                }
            }
        }
    }


# ============================================================================
# Tests for validate_dataset_id
# ============================================================================


class TestValidateDatasetId:
    """Tests for dataset ID validation and normalization."""

    def test_normalizes_numeric_input(self):
        """Numeric input is zero-padded to 6 digits with 'ds' prefix."""
        assert validate_dataset_id("5085") == "ds005085"

    def test_normalizes_ds_prefixed_input(self):
        """Input with 'ds' prefix is normalized to 6-digit format."""
        assert validate_dataset_id("ds5085") == "ds005085"

    def test_already_normalized_input(self):
        """Already normalized input passes through unchanged."""
        assert validate_dataset_id("ds005085") == "ds005085"

    def test_strips_whitespace(self):
        """Whitespace around input is stripped."""
        assert validate_dataset_id("  ds5085  ") == "ds005085"

    def test_uppercase_prefix_accepted(self):
        """Uppercase 'DS' prefix is accepted and normalized."""
        assert validate_dataset_id("DS5085") == "ds005085"

    def test_single_digit_input(self):
        """Single digit input is normalized."""
        assert validate_dataset_id("1") == "ds000001"

    def test_max_numeric_value_accepted(self):
        """Maximum valid numeric value (9999) is accepted."""
        assert validate_dataset_id("9999") == "ds009999"

    def test_zero_padding_preserved(self):
        """Zero padding in input is preserved."""
        assert validate_dataset_id("ds000042") == "ds000042"

    def test_numeric_part_too_large_raises_error(self):
        """Numeric part exceeding 4 digits raises ValueError."""
        with pytest.raises(ValueError, match="too many digits"):
            validate_dataset_id("10000")

    def test_ds_prefix_with_numeric_part_too_large_raises_error(self):
        """'ds' prefix with numeric part too large raises ValueError."""
        with pytest.raises(ValueError, match="too many digits"):
            validate_dataset_id("ds10000")

    def test_non_numeric_suffix_with_ds_prefix_raises_error(self):
        """'ds' prefix with non-numeric suffix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            validate_dataset_id("ds00a00")

    def test_invalid_format_no_digits_raises_error(self):
        """Non-numeric input without 'ds' prefix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            validate_dataset_id("invalid")

    def test_empty_string_raises_error(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            validate_dataset_id("")

    def test_whitespace_only_raises_error(self):
        """Whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset ID format"):
            validate_dataset_id("   ")


# ============================================================================
# Tests for fetch_all_filenames
# ============================================================================


class TestFetchAllFilenames:
    """Tests for fetching all filenames from a dataset."""

    @patch("brainsets.utils.openneuro.openneuro_s3.get_object_list")
    def test_returns_filenames_when_non_empty(self, mock_get_object_list):
        """Returns list of filenames when dataset has files."""
        filenames = [
            "sub-01/ieeg/sub-01_task-VisualNaming_ieeg.edf",
            "sub-01/ieeg/sub-01_task-VisualNaming_channels.tsv",
            "sub-02/ieeg/sub-02_ses-01_task-Rest_acq-ecog_run-01_ieeg.vhdr",
            "sub-02/ieeg/sub-02_ses-01_task-Rest_acq-ecog_run-01_ieeg.vmrk",
            "participants.tsv",
        ]
        mock_get_object_list.return_value = filenames

        result = fetch_all_filenames("ds005085")

        assert result == filenames
        mock_get_object_list.assert_called_once_with(OPENNEURO_S3_BUCKET, "ds005085/")

    @patch("brainsets.utils.openneuro.openneuro_s3.get_object_list")
    def test_normalizes_dataset_id_before_listing(self, mock_get_object_list):
        """Dataset ID is normalized before querying."""
        mock_get_object_list.return_value = ["file.edf"]

        fetch_all_filenames("5085")

        mock_get_object_list.assert_called_once_with(OPENNEURO_S3_BUCKET, "ds005085/")

    @patch("brainsets.utils.openneuro.openneuro_s3.get_object_list")
    def test_raises_runtime_error_on_empty_dataset(self, mock_get_object_list):
        """RuntimeError is raised when no files are found."""
        mock_get_object_list.return_value = []

        with pytest.raises(RuntimeError, match="No files found"):
            fetch_all_filenames("ds005085")

    @patch("brainsets.utils.openneuro.openneuro_s3.get_object_list")
    def test_raises_runtime_error_with_dataset_id_in_message(
        self, mock_get_object_list
    ):
        """RuntimeError message includes the dataset ID."""
        mock_get_object_list.return_value = []

        with pytest.raises(RuntimeError, match="ds005085"):
            fetch_all_filenames("ds005085")


# ============================================================================
# Tests for fetch_participants_tsv
# ============================================================================


class TestFetchParticipantsTsv:
    """Tests for fetching and parsing participants.tsv."""

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_returns_indexed_dataframe_with_participant_id(
        self, mock_get_client, participants_tsv_bytes
    ):
        """Returns DataFrame indexed by participant_id when column exists."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.return_value = {
            "Body": MagicMock(
                read=lambda: participants_tsv_bytes(has_participant_id=True)
            )
        }

        result = fetch_participants_tsv("ds005085")

        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "participant_id"
        assert len(result) == 2
        assert list(result.index) == ["participant_01", "participant_02"]
        assert "age" in result.columns
        assert "sex" in result.columns

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_normalizes_dataset_id_before_fetching(self, mock_get_client):
        """Dataset ID is normalized before S3 fetch."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: b"participant_id\npart_01")
        }

        fetch_participants_tsv("5085")

        call_args = mock_client.get_object.call_args
        assert call_args[1]["Key"] == "ds005085/participants.tsv"

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_returns_none_when_participant_id_column_missing(
        self, mock_get_client, participants_tsv_bytes, caplog
    ):
        """Returns None with warning when participant_id column is absent."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.return_value = {
            "Body": MagicMock(
                read=lambda: participants_tsv_bytes(has_participant_id=False)
            )
        }

        result = fetch_participants_tsv("ds005085")

        assert result is None
        assert "No participant_id column found" in caplog.text

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_returns_none_on_no_such_key_error(self, mock_get_client):
        """Returns None when participants.tsv does not exist (NoSuchKey)."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.side_effect = make_client_error("NoSuchKey")

        result = fetch_participants_tsv("ds005085")

        assert result is None

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_returns_none_on_404_error(self, mock_get_client):
        """Returns None when participants.tsv returns 404."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.side_effect = make_client_error("404")

        result = fetch_participants_tsv("ds005085")

        assert result is None

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_reraises_other_client_errors(self, mock_get_client):
        """Other ClientErrors are re-raised."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.side_effect = make_client_error(
            "AccessDenied", "Access Denied"
        )

        with pytest.raises(ClientError):
            fetch_participants_tsv("ds005085")

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_parses_tsv_with_na_values(self, mock_get_client):
        """TSV parser respects na_values configuration."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        tsv_content = b"participant_id\tage\npart_01\tn/a\npart_02\tN/A"
        mock_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: tsv_content)
        }

        result = fetch_participants_tsv("ds005085")

        assert result is not None
        assert pd.isna(result.loc["part_01", "age"])
        assert pd.isna(result.loc["part_02", "age"])


# ============================================================================
# Tests for construct_s3_url_from_path and download_recording
# ============================================================================


class TestConstructS3UrlFromPath:
    """Tests for S3 URL construction."""

    def test_constructs_url_with_subdirectory(self):
        """URL is correctly constructed from dataset, path, and recording ID."""
        url = construct_s3_url_from_path(
            dataset_id="ds004019",
            data_file_path="sub-01/ses-01/eeg/sub-01_ses-01_task-nap_run-1_eeg.edf",
            recording_id="sub-01_ses-01_task-nap_run-1",
        )

        assert (
            url
            == "s3://openneuro.org/ds004019/sub-01/ses-01/eeg/sub-01_ses-01_task-nap_run-1"
        )

    def test_normalizes_dataset_id(self):
        """Dataset ID is normalized before URL construction."""
        url = construct_s3_url_from_path(
            dataset_id="4019",
            data_file_path="sub-01/eeg/file.edf",
            recording_id="sub-01_task-test",
        )

        assert url.startswith("s3://openneuro.org/ds004019/")

    def test_handles_deeply_nested_path(self):
        """Deeply nested file paths are handled correctly."""
        url = construct_s3_url_from_path(
            dataset_id="ds000001",
            data_file_path="derivatives/sub-A/ses-01/func/deep_file.nii.gz",
            recording_id="recording_x",
        )

        assert "derivatives/sub-A/ses-01/func" in url
        assert url.endswith("recording_x")
        assert (
            url
            == "s3://openneuro.org/ds000001/derivatives/sub-A/ses-01/func/recording_x"
        )


class TestDownloadRecording:
    """Tests for recording download delegation."""

    @patch("brainsets.utils.openneuro.openneuro_s3.download_prefix_from_url")
    def test_delegates_to_download_prefix_from_url(self, mock_download):
        """download_recording delegates to download_prefix_from_url."""
        s3_url = "s3://openneuro.org/ds005085/sub-01/eeg/file"
        target_dir = Path("/tmp/data")
        expected_files = [Path("/tmp/data/file1.edf")]
        mock_download.return_value = expected_files

        result = download_recording(s3_url, target_dir)

        mock_download.assert_called_once_with(s3_url, target_dir)
        assert result == expected_files


# ============================================================================
# Tests for download_dataset_description
# ============================================================================


class TestDownloadDatasetDescription:
    """Tests for dataset_description.json download and caching."""

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_short_circuits_when_file_exists(self, mock_get_client, tmp_path):
        """Returns existing file without downloading if already present."""
        target_file = tmp_path / "dataset_description.json"
        target_file.write_text('{"name": "existing"}')

        result = download_dataset_description("ds005085", tmp_path)

        assert result == target_file
        mock_get_client.assert_not_called()

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_downloads_and_writes_file_on_success(self, mock_get_client, tmp_path):
        """Downloads and writes file when not already present."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: b'{"name": "test"}')
        }

        result = download_dataset_description("ds005085", tmp_path)

        assert result == tmp_path / "dataset_description.json"
        assert result.exists()
        assert result.read_bytes() == b'{"name": "test"}'

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_normalizes_dataset_id_before_fetching(self, mock_get_client, tmp_path):
        """Dataset ID is normalized before S3 fetch."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.return_value = {"Body": MagicMock(read=lambda: b"{}")}

        download_dataset_description("5085", tmp_path)

        call_args = mock_client.get_object.call_args
        assert call_args[1]["Key"] == "ds005085/dataset_description.json"

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_creates_parent_directories(self, mock_get_client, tmp_path):
        """Creates parent directories if they don't exist."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.return_value = {"Body": MagicMock(read=lambda: b"{}")}
        nested_target = tmp_path / "a" / "b" / "c"

        result = download_dataset_description("ds005085", nested_target)

        assert result.parent.exists()
        assert result.exists()

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_raises_runtime_error_on_no_such_key(self, mock_get_client, tmp_path):
        """RuntimeError is raised when file doesn't exist on S3 (NoSuchKey)."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.side_effect = make_client_error("NoSuchKey")

        with pytest.raises(RuntimeError, match="not found"):
            download_dataset_description("ds005085", tmp_path)

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_raises_runtime_error_on_404(self, mock_get_client, tmp_path):
        """RuntimeError is raised on 404 error."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.side_effect = make_client_error("404")

        with pytest.raises(RuntimeError, match="not found"):
            download_dataset_description("ds005085", tmp_path)

    @patch("brainsets.utils.openneuro.openneuro_s3.get_cached_s3_client")
    def test_raises_runtime_error_on_other_client_error(
        self, mock_get_client, tmp_path
    ):
        """RuntimeError is raised for other ClientErrors."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_object.side_effect = make_client_error(
            "AccessDenied", "Access Denied"
        )

        with pytest.raises(RuntimeError, match="Failed to download"):
            download_dataset_description("ds005085", tmp_path)


# ============================================================================
# Tests for validate_dataset_version
# ============================================================================


class TestValidateDatasetVersion:
    """Tests for dataset version validation."""

    @patch("brainsets.utils.openneuro.openneuro_s3._graphql_query_openneuro")
    def test_returns_latest_tag_when_versions_match(self, mock_graphql):
        """Returns latest tag when provided version matches."""
        mock_graphql.return_value = graphql_ok_response("1.0.0")

        result = validate_dataset_version("ds005085", "1.0.0")

        assert result == "1.0.0"

    @patch("brainsets.utils.openneuro.openneuro_s3._graphql_query_openneuro")
    def test_warns_when_versions_differ(self, mock_graphql, caplog):
        """Logs warning when provided version differs from latest."""
        mock_graphql.return_value = graphql_ok_response("2.0.0")

        result = validate_dataset_version("ds005085", "1.0.0")

        assert result == "2.0.0"
        assert "version '1.0.0' was used to create the brainset pipeline" in caplog.text
        assert "but the latest available version on OpenNeuro is '2.0.0'" in caplog.text

    @patch("brainsets.utils.openneuro.openneuro_s3._graphql_query_openneuro")
    def test_queries_with_correct_variables(self, mock_graphql):
        """GraphQL is queried with the correct dataset ID."""
        mock_graphql.return_value = graphql_ok_response("1.0.0")

        validate_dataset_version("ds005085", "1.0.0")

        call_args = mock_graphql.call_args
        assert call_args[0][1]["datasetId"] == "ds005085"

    @patch("brainsets.utils.openneuro.openneuro_s3._graphql_query_openneuro")
    def test_does_not_warn_when_versions_match(self, mock_graphql, caplog):
        """No warning is logged when versions match."""
        mock_graphql.return_value = graphql_ok_response("1.0.0")

        validate_dataset_version("ds005085", "1.0.0")

        # Caplog will contain only debug logs and errors, not warnings
        for record in caplog.records:
            if record.levelno >= 30:  # WARNING level and above
                assert "version" not in record.message.lower()


# ============================================================================
# Tests for _graphql_query_openneuro
# ============================================================================


class TestGraphqlQueryOpenneuro:
    """Tests for GraphQL query execution and retry logic."""

    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_succeeds_on_200_status(self, mock_post):
        """Returns response JSON on 200 status code."""
        expected_response = {"data": {"test": "value"}}
        mock_post.return_value = MagicMock(
            status_code=200, json=lambda: expected_response
        )

        result = _graphql_query_openneuro("query { test }", {})

        assert result == expected_response

    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_raises_on_non_200_status(self, mock_post):
        """Raises exception on non-200 status code."""
        mock_post.return_value = MagicMock(status_code=500)

        with pytest.raises(Exception, match="Query failed"):
            _graphql_query_openneuro("query { test }", {})

    @patch("time.sleep")
    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_retries_on_transient_failure_then_succeeds(self, mock_post, mock_sleep):
        """Retries on transient failure and succeeds on retry."""
        expected_response = {"data": {"test": "value"}}
        mock_post.side_effect = [
            Exception("Network error"),
            MagicMock(status_code=200, json=lambda: expected_response),
        ]

        result = _graphql_query_openneuro("query { test }", {})

        assert result == expected_response
        assert mock_post.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("time.sleep")
    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_raises_after_max_attempts(self, mock_post, mock_sleep):
        """Raises exception after exhausting retry attempts."""
        mock_post.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            _graphql_query_openneuro("query { test }", {})

        # 5 attempts total (initial + 4 retries)
        assert mock_post.call_count == 5
        # 4 sleeps between retries
        assert mock_sleep.call_count == 4

    @patch("time.sleep")
    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_exponential_backoff_timing(self, mock_post, mock_sleep):
        """Retry delays follow exponential backoff pattern."""
        mock_post.side_effect = Exception("Network error")

        try:
            _graphql_query_openneuro("query { test }", {})
        except Exception:
            pass

        # Sleep calls should follow exponential backoff: 4, 8, 10, 10
        # (capped at max_wait=10)
        sleep_calls = [call_args[0][0] for call_args in mock_sleep.call_args_list]
        assert sleep_calls == [4, 8, 10, 10]

    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_passes_query_and_variables_correctly(self, mock_post):
        """Query and variables are passed correctly to requests.post."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {})
        query = "query { test }"
        variables = {"id": "123"}

        _graphql_query_openneuro(query, variables)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["query"] == query
        assert call_kwargs["json"]["variables"] == variables

    @patch("brainsets.utils.openneuro.openneuro_s3.requests.post")
    def test_handles_none_variables(self, mock_post):
        """None variables are handled correctly."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {})

        _graphql_query_openneuro("query { test }", None)

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["variables"] is None
