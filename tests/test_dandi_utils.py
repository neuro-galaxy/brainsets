from types import SimpleNamespace

import numpy as np
import pytest

from brainsets.taxonomy import Hemisphere, Sex, Species
from brainsets.utils.dandi_utils import extract_ecog_from_nwb, extract_subject_from_nwb


class FakeElectrodes:
    def __init__(self, **columns):
        self._columns = {name: np.asarray(values) for name, values in columns.items()}
        self.colnames = list(self._columns.keys())
        for name, values in self._columns.items():
            setattr(self, name, values)

    def __getitem__(self, key):
        return self._columns[key]


def _build_nwb_for_ecog(data, rate, electrodes, subject=None):
    electrical_series = SimpleNamespace(data=np.asarray(data), rate=rate)
    return SimpleNamespace(
        acquisition={"ElectricalSeries": electrical_series},
        electrodes=electrodes,
        subject=subject,
    )


class TestExtractSubjectFromNwb:

    def test_extract_subject_normalizes_id_species_and_sex(self):
        nwbfile = SimpleNamespace(
            subject=SimpleNamespace(
                subject_id=" Sub-01 ",
                species="NCBITaxon_9541",
                sex="f",
            )
        )

        subject = extract_subject_from_nwb(nwbfile)

        assert subject.id == "sub-01"
        assert subject.species == Species.MACACA_FASCICULARIS
        assert subject.sex == Sex.FEMALE

    def test_extract_subject_falls_back_to_unknown_for_blank_fields(self):
        nwbfile = SimpleNamespace(
            subject=SimpleNamespace(subject_id="sub-02", species=" ", sex=None)
        )

        subject = extract_subject_from_nwb(nwbfile)

        assert subject.id == "sub-02"
        assert subject.species == Species.UNKNOWN
        assert subject.sex == Sex.UNKNOWN


class TestExtractEcogFromNwb:

    def test_extract_ecog_infers_hemisphere_from_electrodes_and_marks_bad_channels(self):
        signal = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            dtype=np.float64,
        )
        electrodes = FakeElectrodes(
            location=["left temporal", "left frontal", "left motor"],
            group_name=["grid", "depth", "grid"],
            good=[True, False, True],
        )
        nwbfile = _build_nwb_for_ecog(signal, rate=100.0, electrodes=electrodes)

        ecog, channels = extract_ecog_from_nwb(nwbfile)

        np.testing.assert_allclose(np.asarray(ecog.signal), signal)
        assert ecog.sampling_rate == 100.0
        np.testing.assert_allclose(np.asarray(ecog.domain.start), np.array([0.0]))
        np.testing.assert_allclose(np.asarray(ecog.domain.end), np.array([0.02]))
        np.testing.assert_array_equal(
            np.asarray(channels.hemisphere),
            np.full(3, int(Hemisphere.LEFT)),
        )
        np.testing.assert_array_equal(np.asarray(channels.bad), np.array([False, True, False]))
        np.testing.assert_array_equal(
            np.asarray(channels.group),
            np.array(["grid", "depth", "grid"]),
        )

    def test_extract_ecog_prefers_explicit_hemisphere_over_inferred_values(self):
        signal = np.array([[1.0, 2.0], [3.0, 4.0]])
        electrodes = FakeElectrodes(
            location=["left", "left"],
            group_name=["g1", "g2"],
            good=[True, True],
        )
        nwbfile = _build_nwb_for_ecog(signal, rate=50.0, electrodes=electrodes)

        _, channels = extract_ecog_from_nwb(nwbfile, subject_hemisphere="R")

        np.testing.assert_array_equal(
            np.asarray(channels.hemisphere),
            np.full(2, int(Hemisphere.RIGHT)),
        )

    def test_extract_ecog_uses_subject_metadata_when_electrodes_are_ambiguous(self):
        signal = np.array([[1.0], [2.0], [3.0]])
        electrodes = FakeElectrodes(
            location=["left and right"],
            group_name=["g1"],
            good=[True],
        )
        nwbfile = _build_nwb_for_ecog(
            signal,
            rate=25.0,
            electrodes=electrodes,
            subject=SimpleNamespace(hemisphere="left"),
        )

        _, channels = extract_ecog_from_nwb(nwbfile)

        np.testing.assert_array_equal(
            np.asarray(channels.hemisphere),
            np.array([int(Hemisphere.LEFT)]),
        )

    def test_extract_ecog_raises_when_electrical_series_is_missing(self):
        nwbfile = SimpleNamespace(acquisition={}, electrodes=FakeElectrodes())

        with pytest.raises(KeyError, match="ElectricalSeries"):
            extract_ecog_from_nwb(nwbfile)
