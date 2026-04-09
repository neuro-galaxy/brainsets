"""Unit tests for S3 utility functions."""

import pytest
from unittest.mock import MagicMock, patch
from brainsets.utils.s3_utils import BOTO_AVAILABLE

pytestmark = pytest.mark.skipif(
    not BOTO_AVAILABLE, reason="boto3/botocore not installed"
)

from brainsets.utils.s3_utils import (
    UNSIGNED,
    ClientError,
    clear_cached_s3_client,
    download_public_object,
    fetch_public_object,
    get_cached_s3_client,
    get_object_list,
    download_prefix,
    download_prefix_from_url,
)


class TestGetCachedS3Client:
    def setup_method(self):
        clear_cached_s3_client()

    def teardown_method(self):
        clear_cached_s3_client()

    @patch("brainsets.utils.s3_utils.boto3.client")
    def test_returns_s3_client(self, mock_boto_client):
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        result = get_cached_s3_client()

        assert result == mock_client
        mock_boto_client.assert_called_once()
        call_args = mock_boto_client.call_args
        assert call_args[0][0] == "s3"

    @patch("brainsets.utils.s3_utils.boto3.client")
    def test_config_uses_unsigned_signature(self, mock_boto_client):
        get_cached_s3_client()

        call_args = mock_boto_client.call_args
        config = call_args[1]["config"]
        assert config.signature_version == UNSIGNED

    @patch("brainsets.utils.s3_utils.boto3.client")
    def test_config_with_custom_retry_mode(self, mock_boto_client):
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        get_cached_s3_client(
            retry_mode="standard", max_attempts=3, max_pool_connections=20
        )

        call_args = mock_boto_client.call_args
        config = call_args[1]["config"]
        assert config.retries["mode"] == "standard"
        assert config.retries["total_max_attempts"] == 3
        assert config.max_pool_connections == 20

    @patch("brainsets.utils.s3_utils.boto3.client")
    def test_config_with_adaptive_retry_mode(self, mock_boto_client):
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        get_cached_s3_client(retry_mode="adaptive", max_attempts=10)

        call_args = mock_boto_client.call_args
        config = call_args[1]["config"]
        assert config.retries["mode"] == "adaptive"
        assert config.retries["total_max_attempts"] == 10

    @patch("brainsets.utils.s3_utils.boto3.client")
    def test_raises_on_different_params(self, mock_boto_client):
        mock_boto_client.return_value = MagicMock()

        get_cached_s3_client(retry_mode="adaptive", max_attempts=5)

        with pytest.raises(ValueError, match="already called with"):
            get_cached_s3_client(retry_mode="standard", max_attempts=10)


class TestListObjects:
    def test_lists_objects_successfully(self):
        """Test listing objects returns keys without directories."""
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "ds000000/subdir/file1.edf"},
                    {"Key": "ds000000/subdir/subsubdir/"},
                    {"Key": "ds000000/subdir/file2.edf"},
                    {"Key": "ds000000/subdir/subsubdir/file3.edf"},
                ]
            }
        ]

        result = get_object_list("test-bucket", "ds000000/", s3_client=mock_client)

        assert result == [
            "subdir/file1.edf",
            "subdir/file2.edf",
            "subdir/subsubdir/file3.edf",
        ]
        mock_client.get_paginator.assert_called_once_with("list_objects_v2")
        mock_paginator.paginate.assert_called_once_with(
            Bucket="test-bucket", Prefix="ds000000/"
        )

    def test_handles_empty_bucket(self):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]

        result = get_object_list("test-bucket", "empty/", s3_client=mock_client)

        assert result == []

    def test_handles_multiple_pages(self):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "prefix/file1.edf"}]},
            {"Contents": [{"Key": "prefix/file2.edf"}]},
        ]

        result = get_object_list("test-bucket", "prefix/", s3_client=mock_client)

        assert result == ["file1.edf", "file2.edf"]

    def test_raises_runtime_error_on_failure(self):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="Error listing objects"):
            get_object_list("test-bucket", "prefix/", s3_client=mock_client)

    @patch("brainsets.utils.s3_utils.get_cached_s3_client")
    def test_uses_cached_client_when_none_provided(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]

        get_object_list("test-bucket", "prefix/")

        mock_get_client.assert_called_once()

    def test_handles_key_not_starting_with_prefix(self):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "ds000000/file1.edf"},
                    {"Key": "other/file2.edf"},
                ]
            }
        ]
        result = get_object_list("test-bucket", "ds000000/", s3_client=mock_client)
        assert "file1.edf" in result
        assert "other/file2.edf" not in result

    def test_filters_empty_relative_path(self):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "ds000000"},
                    {"Key": "ds000000/file1.edf"},
                ]
            }
        ]
        result = get_object_list("test-bucket", "ds000000", s3_client=mock_client)

        assert "/file1.edf" in result
        assert "" not in result
        assert len(result) == 1


class TestFetchPublicObject:
    def test_returns_object_body(self):
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"file is loaded"
        mock_client.get_object.return_value = {"Body": mock_body}

        result = fetch_public_object(
            "my-bucket", "path/to/object.txt", s3_client=mock_client
        )

        assert result == b"file is loaded"
        mock_client.get_object.assert_called_once_with(
            Bucket="my-bucket", Key="path/to/object.txt"
        )

    def test_raises_file_not_found_when_missing(self):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject"
        )

        with pytest.raises(FileNotFoundError, match="my-bucket/path/to/key"):
            fetch_public_object("my-bucket", "path/to/key", s3_client=mock_client)

    def test_raises_file_not_found_on_404_code(self):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject"
        )

        with pytest.raises(FileNotFoundError):
            fetch_public_object("b", "k", s3_client=mock_client)

    def test_raises_runtime_error_on_other_client_error(self):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "GetObject"
        )

        with pytest.raises(RuntimeError, match="Failed to fetch"):
            fetch_public_object("b", "k", s3_client=mock_client)

    @patch("brainsets.utils.s3_utils.get_cached_s3_client")
    def test_uses_cached_client_when_none_provided(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_body = MagicMock()
        mock_body.read.return_value = b"x"
        mock_client.get_object.return_value = {"Body": mock_body}

        fetch_public_object("bucket", "key")

        mock_get_client.assert_called_once()
        mock_client.get_object.assert_called_once()


class TestDownloadPublicObject:
    def test_writes_file(self, tmp_path):
        mock_client = MagicMock()
        dest = tmp_path / "nested" / "out.txt"

        result = download_public_object(
            "my-bucket", "path/key.txt", dest, s3_client=mock_client
        )

        assert result == dest
        mock_client.download_file.assert_called_once_with(
            "my-bucket", "path/key.txt", str(dest)
        )
        assert dest.parent.is_dir()

    def test_raises_file_not_found_when_missing(self, tmp_path):
        mock_client = MagicMock()
        mock_client.download_file.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject"
        )
        dest = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError):
            download_public_object("b", "k", dest, s3_client=mock_client)

        assert not dest.exists()

    def test_removes_partial_file_on_failure(self, tmp_path):
        from pathlib import Path

        mock_client = MagicMock()

        def fail_download(*args, **kwargs):
            dest = Path(args[2])
            dest.write_bytes(b"partial")
            raise ClientError(
                {"Error": {"Code": "403", "Message": "Forbidden"}}, "GetObject"
            )

        mock_client.download_file.side_effect = fail_download
        dest = tmp_path / "out.bin"

        with pytest.raises(RuntimeError, match="Failed to download"):
            download_public_object("b", "k", dest, s3_client=mock_client)

        assert not dest.exists()

    @patch("brainsets.utils.s3_utils.get_cached_s3_client")
    def test_uses_cached_client_when_none_provided(self, mock_get_client, tmp_path):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        dest = tmp_path / "f.txt"

        download_public_object("bucket", "key", dest)

        mock_get_client.assert_called_once()
        mock_client.download_file.assert_called_once()


class TestDownloadPrefix:
    def test_downloads_files_successfully(self, tmp_path):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "ds000000/sub-1/file1.edf"},
                    {"Key": "ds000000/sub-1/file2.edf"},
                ]
            }
        ]

        result = download_prefix(
            bucket="test-bucket",
            prefix="ds000000/sub-1/",
            target_dir=tmp_path,
            s3_client=mock_client,
        )

        assert len(result) == 2
        assert tmp_path / "sub-1" / "file1.edf" in result
        assert tmp_path / "sub-1" / "file2.edf" in result
        assert mock_client.download_file.call_count == 2

    def test_strips_custom_prefix(self, tmp_path):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "data/Projects/EEG/subject1/file.edf"}]}
        ]

        result = download_prefix(
            bucket="test-bucket",
            prefix="data/Projects/EEG/",
            target_dir=tmp_path,
            strip_prefix="data/Projects/",
            s3_client=mock_client,
        )

        assert len(result) == 1
        expected_path = tmp_path / "EEG" / "subject1" / "file.edf"
        assert result[0] == expected_path

    def test_skips_directories(self, tmp_path):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "ds000000/subdir/"},
                    {"Key": "ds000000/subdir/file1.edf"},
                    {"Key": "ds000000/file2.edf"},
                    {"Key": "ds000000/subdir/subsubdir/file3.edf"},
                ]
            }
        ]

        result = download_prefix(
            bucket="test-bucket",
            prefix="ds000000/",
            target_dir=tmp_path,
            s3_client=mock_client,
        )

        assert len(result) == 3
        assert mock_client.download_file.call_count == 3

    def test_raises_error_when_no_files_found(self, tmp_path):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]

        with pytest.raises(RuntimeError, match="No files found"):
            download_prefix(
                bucket="test-bucket",
                prefix="nonexistent/",
                target_dir=tmp_path,
                s3_client=mock_client,
            )

    def test_raises_error_on_download_failure(self, tmp_path):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "ds000000/file.edf"}]}
        ]
        mock_client.download_file.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Access Denied"}}, "GetObject"
        )

        with pytest.raises(RuntimeError, match="Failed to download"):
            download_prefix(
                bucket="test-bucket",
                prefix="ds000000/",
                target_dir=tmp_path,
                s3_client=mock_client,
            )

    def test_creates_parent_directories(self, tmp_path):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "ds000000/deep/nested/path/file.edf"}]}
        ]

        download_prefix(
            bucket="test-bucket",
            prefix="ds000000/",
            target_dir=tmp_path,
            s3_client=mock_client,
        )

        expected_dir = tmp_path / "deep" / "nested" / "path"
        assert expected_dir.exists()

    def test_handles_prefix_without_subdirectory(self, tmp_path):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "ds000000/file.edf"}]}
        ]

        result = download_prefix(
            bucket="test-bucket",
            prefix="ds000000",
            target_dir=tmp_path,
            s3_client=mock_client,
        )

        assert len(result) == 1

    def test_handles_obj_key_not_starting_with_strip_prefix(self, tmp_path):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "other/path/file.edf"}]}
        ]
        result = download_prefix(
            bucket="test-bucket",
            prefix="ds000000/",
            target_dir=tmp_path,
            strip_prefix="ds000000/",
            s3_client=mock_client,
        )

        assert len(result) == 1
        expected_path = tmp_path / "other" / "path" / "file.edf"
        assert result[0] == expected_path

    def test_raises_error_on_general_exception(self, tmp_path):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = ValueError("Unexpected error")

        with pytest.raises(RuntimeError, match="Failed to download from"):
            download_prefix(
                bucket="test-bucket",
                prefix="ds000000/",
                target_dir=tmp_path,
                s3_client=mock_client,
            )

    @patch("brainsets.utils.s3_utils.get_cached_s3_client")
    def test_uses_cached_client_when_none_provided(self, mock_get_client, tmp_path):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "ds000000/file.edf"}]}
        ]

        download_prefix(
            bucket="test-bucket",
            prefix="ds000000/",
            target_dir=tmp_path,
        )

        mock_get_client.assert_called_once()


class TestDownloadPrefixFromUrl:
    @patch("brainsets.utils.s3_utils.download_prefix")
    def test_parses_s3_url_correctly(self, mock_download, tmp_path):
        mock_download.return_value = [tmp_path / "file.edf"]

        result = download_prefix_from_url(
            s3_url="s3://my-bucket/path/to/files/",
            target_dir=tmp_path,
        )

        mock_download.assert_called_once_with("my-bucket", "path/to/files/", tmp_path)
        assert result == [tmp_path / "file.edf"]

    @patch("brainsets.utils.s3_utils.download_prefix")
    def test_handles_url_without_trailing_slash(self, mock_download, tmp_path):
        mock_download.return_value = []

        download_prefix_from_url(
            s3_url="s3://bucket/prefix",
            target_dir=tmp_path,
        )

        mock_download.assert_called_once_with("bucket", "prefix", tmp_path)

    def test_raises_error_for_invalid_scheme(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid S3 URL"):
            download_prefix_from_url(
                s3_url="https://bucket.s3.amazonaws.com/path",
                target_dir=tmp_path,
            )

    def test_raises_error_for_http_url(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid S3 URL"):
            download_prefix_from_url(
                s3_url="http://example.com/file.edf",
                target_dir=tmp_path,
            )

    @patch("brainsets.utils.s3_utils.download_prefix")
    def test_strips_leading_slash_from_prefix(self, mock_download, tmp_path):
        mock_download.return_value = []

        download_prefix_from_url(
            s3_url="s3://bucket/path/to/data",
            target_dir=tmp_path,
        )

        # Prefix should be "path/to/data", not "/path/to/data"
        call_args = mock_download.call_args[0]
        assert call_args[1] == "path/to/data"
        assert not call_args[1].startswith("/")

    @patch("brainsets.utils.s3_utils.download_prefix")
    def test_handles_url_with_leading_slash_in_path(self, mock_download, tmp_path):
        mock_download.return_value = []

        download_prefix_from_url(
            s3_url="s3://bucket//path/to/data",
            target_dir=tmp_path,
        )
        call_args = mock_download.call_args[0]
        assert call_args[1] == "path/to/data"
        assert not call_args[1].startswith("/")
