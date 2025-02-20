import pytest
from brainsets.descriptions import (
    BrainsetDescription,
)  # Replace `your_module` with the actual module name


def test_brainset_description_creation():
    brainset_description = BrainsetDescription(
        id="data_id",
        origin_version="dandi_id",
        derived_version="version",
        source="source",
        description=(
            "This dataset has neural activity and behavior"
            " this is another line of description"
        ),
        publication_list=["Some publication reference", "another_reference"],
        subject_list=["bob", "sue"],
    )

    # Check attributes
    assert brainset_description.id == "data_id"
    assert brainset_description.origin_version == "dandi_id"
    assert brainset_description.derived_version == "version"
    assert brainset_description.source == "source"
    assert (
        "This dataset has neural activity and behavior"
        in brainset_description.description
    )
    assert brainset_description.publication_list == [
        "Some publication reference",
        "another_reference",
    ]
    assert brainset_description.subject_list == ["bob", "sue"]
    assert isinstance(brainset_description.brainsets_version, str)
    assert isinstance(brainset_description.temporaldata_version, str)
