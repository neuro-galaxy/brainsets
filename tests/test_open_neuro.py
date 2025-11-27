from brainsets.utils.open_neuro import (
    validate_dataset_id,
    fetch_metadata,
    fetch_readme,
    fetch_latest_version_tag,
    fetch_all_version_tags,
    fetch_all_filenames,
    fetch_participants,
    download_file_from_s3,
    get_s3_file_size,
)
import pytest
import tempfile
import json
import pandas as pd
from pathlib import Path


def test_validate_dataset_id():
    assert validate_dataset_id("ds006695") == "ds006695"
    assert validate_dataset_id("5085") == "ds005085"
    assert validate_dataset_id("ds5085") == "ds005085"
    assert validate_dataset_id("ds005085") == "ds005085"

    with pytest.raises(ValueError):
        validate_dataset_id("ds0050851")
    with pytest.raises(ValueError):
        validate_dataset_id("50851")
    with pytest.raises(ValueError):
        validate_dataset_id("ds0066951")
    with pytest.raises(ValueError):
        validate_dataset_id("50851")


@pytest.mark.parametrize("dataset_id, error", [("ds006695", False), ("ds00555", True)])
def test_get_metadata(dataset_id, error):
    if error:
        with pytest.raises(RuntimeError):
            fetch_metadata(dataset_id)
    else:
        metadata = fetch_metadata(dataset_id)
        assert metadata is not None
        assert metadata["datasetId"] == dataset_id
        assert metadata["created"] is not None
        assert metadata["datasetName"] is not None


@pytest.mark.parametrize("dataset_id, error", [("ds006695", False), ("ds00555", True)])
def test_get_readme(dataset_id, error):
    if error:
        with pytest.raises(RuntimeError):
            fetch_readme(dataset_id)
    else:
        readme = fetch_readme(dataset_id)
        assert readme is not None
        assert isinstance(readme, str)


@pytest.mark.parametrize("dataset_id, error", [("ds006695", False), ("ds00555", True)])
def test_fetch_latest_version_tag(dataset_id, error):
    if error:
        with pytest.raises(RuntimeError):
            fetch_latest_version_tag(dataset_id)
    else:
        version_tag = fetch_latest_version_tag(dataset_id)
        assert version_tag is not None
        assert isinstance(version_tag, str)
        # Version tags should follow semantic versioning pattern (e.g., "1.0.0")
        assert len(version_tag.split(".")) >= 2


@pytest.mark.parametrize("dataset_id, error", [("ds006695", False), ("ds00555", True)])
def test_fetch_all_version_tags(dataset_id, error):
    if error:
        with pytest.raises(RuntimeError):
            fetch_all_version_tags(dataset_id)
    else:
        version_tags = fetch_all_version_tags(dataset_id)
        assert version_tags is not None
        assert isinstance(version_tags, list)
        assert len(version_tags) > 0
        # All elements should be strings
        for tag in version_tags:
            assert isinstance(tag, str)
            # Version tags should follow semantic versioning pattern
            assert len(tag.split(".")) >= 2


@pytest.mark.parametrize("dataset_id, error", [("ds006695", False), ("ds00555", True)])
def test_fetch_all_filenames(dataset_id, error):
    if error:
        with pytest.raises(RuntimeError):
            fetch_all_filenames(dataset_id)
    else:
        filenames = fetch_all_filenames(dataset_id)
        assert filenames is not None
        assert isinstance(filenames, list)
        assert len(filenames) > 0

        has_nested_files = False

        for filename in filenames:
            assert isinstance(filename, str)
            assert len(filename) > 0

            assert not filename.endswith("/"), f"Found directory in results: {filename}"

            assert not filename.startswith(
                dataset_id
            ), f"Path should be relative, not include dataset_id: {filename}"

            if "/" in filename:
                has_nested_files = True

        assert (
            has_nested_files
        ), "Should include files in subdirectories, not just root-level files"


def test_fetch_all_filenames_with_tag():
    """Test fetching filenames with an explicit tag parameter."""
    dataset_id = "ds006695"

    all_tags = fetch_all_version_tags(dataset_id)
    assert len(all_tags) > 0, "Dataset should have at least one version tag"

    tag = all_tags[0]

    filenames = fetch_all_filenames(dataset_id, tag=tag)
    assert filenames is not None
    assert isinstance(filenames, list)
    assert len(filenames) > 0

    for filename in filenames:
        assert isinstance(filename, str)
        assert len(filename) > 0
        assert not filename.endswith("/")
        assert not filename.startswith(dataset_id)
        assert not tag in filename.split("/")[0] if "/" in filename else True


@pytest.mark.parametrize("dataset_id, error", [("ds006695", False), ("ds00555", True)])
def test_fetch_participants(dataset_id, error):
    if error:
        with pytest.raises(RuntimeError):
            fetch_participants(dataset_id)
    else:
        participants = fetch_participants(dataset_id)
        assert participants is not None
        assert isinstance(participants, list)
        assert len(participants) > 0
        # All elements should be strings with 'sub-' prefix
        for participant in participants:
            assert isinstance(participant, str)
            assert participant.startswith("sub-")
            assert len(participant) > 4  # At least "sub-" + some ID


def test_get_s3_file_size():
    """Test getting file size from S3 without downloading."""
    dataset_id = "ds006695"
    file_path = "dataset_description.json"

    file_size = get_s3_file_size(dataset_id, file_path)

    assert file_size > 0, "File size should be greater than 0"
    assert isinstance(file_size, int), "File size should be an integer"


def test_get_s3_file_size_invalid():
    """Test getting file size for non-existent file."""
    dataset_id = "ds006695"
    file_path = "non_existent_file.json"

    with pytest.raises(RuntimeError):
        get_s3_file_size(dataset_id, file_path)


def test_download_file_from_s3():
    """Test downloading a single file from S3."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_id = "ds006695"
        file_path = "dataset_description.json"

        local_path = download_file_from_s3(dataset_id, file_path, temp_dir)

        assert Path(local_path).exists(), "Downloaded file should exist"
        assert Path(local_path).is_file(), "Downloaded path should be a file"

        with open(local_path, "r") as f:
            data = json.load(f)

        assert isinstance(data, dict), "JSON file should contain a dictionary"
        assert len(data) > 0, "JSON file should not be empty"


def test_download_file_from_s3_caching():
    """Test that downloading the same file twice uses caching."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_id = "ds006695"
        file_path = "dataset_description.json"

        local_path1 = download_file_from_s3(dataset_id, file_path, temp_dir)

        mtime1 = Path(local_path1).stat().st_mtime

        local_path2 = download_file_from_s3(dataset_id, file_path, temp_dir)
        mtime2 = Path(local_path2).stat().st_mtime

        assert local_path1 == local_path2, "Same file path should be returned"
        assert (
            mtime1 == mtime2
        ), "File should not be re-downloaded (same modification time)"


def test_download_file_from_s3_invalid():
    """Test downloading non-existent file from S3."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_id = "ds006695"
        file_path = "non_existent_file.json"

        with pytest.raises(RuntimeError):
            download_file_from_s3(dataset_id, file_path, temp_dir)


# TODO: Implement this test
def test_download_subject_eeg_data():
    pass


# TODO: Implement this test
def test_download_openneuro_data():
    pass
