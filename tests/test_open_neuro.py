from brainsets.utils.open_neuro import (
    validate_dataset_id,
    fetch_metadata,
    fetch_readme,
    fetch_latest_version_tag,
    fetch_all_version_tags,
    fetch_all_filenames,
    fetch_participants,
    download_configuration_files,
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
        # All elements should be strings representing file paths
        for filename in filenames:
            assert isinstance(filename, str)
            assert len(filename) > 0


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


def test_download_configuration_files():
    """Test downloading configuration files to a temporary directory and verify their format."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download configuration files
        download_configuration_files(dataset_id="ds006695", target_dir=temp_dir)

        # Check that the temporary directory contains downloaded files
        temp_path = Path(temp_dir)
        downloaded_files = list(temp_path.rglob("*"))

        # Filter out directories, only keep files
        downloaded_files = [f for f in downloaded_files if f.is_file()]

        # Assert that files were downloaded
        assert len(downloaded_files) > 0, "No files were downloaded"

        # Check for expected file types (JSON and TSV files)
        json_files = [f for f in downloaded_files if f.suffix == ".json"]
        tsv_files = [f for f in downloaded_files if f.suffix == ".tsv"]

        # Should have at least some configuration files
        assert (
            len(json_files) > 0 or len(tsv_files) > 0
        ), "No JSON or TSV configuration files found"

        # Validate JSON files are properly formatted
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    json.load(f)  # Should parse without error
                assert (
                    json_file.stat().st_size > 0
                ), f"JSON file {json_file.name} is empty"
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON format in file: {json_file.name}")

        # Validate TSV files are properly formatted
        for tsv_file in tsv_files:
            try:
                df = pd.read_csv(tsv_file, sep="\t")
                assert not df.empty, f"TSV file {tsv_file.name} is empty"
                assert len(df.columns) > 0, f"TSV file {tsv_file.name} has no columns"
            except Exception as e:
                pytest.fail(f"Invalid TSV format in file {tsv_file.name}: {str(e)}")

        # Verify that excluded files are not present
        excluded_patterns = ["events", "_T1w", "scans"]
        for file in downloaded_files:
            filename_lower = file.name.lower()
            for pattern in excluded_patterns:
                assert (
                    pattern not in filename_lower
                ), f"Excluded file pattern '{pattern}' found in {file.name}"


# TODO: Implement this test
def test_download_subject_eeg_data():
    pass


# TODO: Implement this test
def test_download_openneuro_data():
    pass
