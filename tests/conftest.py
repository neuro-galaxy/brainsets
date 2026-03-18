"""Shared pytest fixtures for brainsets tests."""

from pathlib import Path

import pytest


@pytest.fixture
def bids_root(tmp_path):
    """Create a temporary BIDS root directory.

    This is a generic fixture for tests that need a temporary BIDS-compliant
    directory structure. Use this as a base for other fixtures that need to
    populate the directory with test data.

    Returns:
        Path: Path to the temporary BIDS root directory.
    """
    return tmp_path


@pytest.fixture
def bids_root_with_participants(bids_root):
    """Create a BIDS root directory with a participants.tsv file.

    Includes three sample participants with varied demographic data:
    - sub-01: age=34, sex=F
    - sub-02: age=n/a (missing), sex=N/A (missing)
    - sub-03: age=28, sex=M

    Args:
        bids_root: Temporary BIDS root directory fixture.

    Returns:
        Path: Path to the BIDS root directory with participants.tsv.
    """
    participants_tsv = bids_root / "participants.tsv"
    participants_tsv.write_text(
        "participant_id\tage\tsex\n"
        "sub-01\t34\tF\n"
        "sub-02\tn/a\tN/A\n"
        "sub-03\t28\tM\n"
    )
    return bids_root
