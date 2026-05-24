import pytest
from brainsets.descriptions import SubjectDescription


class TestSubjectDescription:

    def test_basic_usage_with_all_parameters(self):
        result = SubjectDescription(
            id="subject_1",
            age=30.5,
            sex="MALE",
            species="HUMAN",
            extra_md="something-extra",
        )

        assert isinstance(result, SubjectDescription)
        assert result.id == "subject_1"
        assert result.age == 30.5
        assert result.sex == "MALE"
        assert result.species == "HUMAN"
        assert result.extra_md == "something-extra"

    def test_minimal_usage_with_only_id(self):
        result = SubjectDescription(id="subject_1")

        assert isinstance(result, SubjectDescription)
        assert result.id == "subject_1"
        assert result.age == None
        assert result.sex == None
        assert result.species == None

    # Age normalization tests
    def test_age_as_int(self):
        result = SubjectDescription(id="subject_1", age=25)
        assert result.age == 25.0
        assert isinstance(result.age, float)

    def test_age_as_float(self):
        result = SubjectDescription(id="subject_1", age=30.5)
        assert result.age == 30.5

    def test_age_as_string_numeric(self):
        result = SubjectDescription(id="subject_1", age="45.7")
        assert result.age == 45.7

    def test_invalid_age_raises(self):
        with pytest.raises(ValueError):
            SubjectDescription(id="subject_1", age="invalid")

    def test_sex_type_validation(self):
        invalid_options = [True, 0.1, 0, {1, 2}, (1, 2), [1, 2]]
        for sex in invalid_options:
            with pytest.raises(ValueError, match="sex must be a string or None"):
                SubjectDescription(id="subject_1", sex=sex)  # type: ignore

        with pytest.raises(ValueError, match="sex cannot be an empty string"):
            SubjectDescription(id="subject_1", sex="")

    def test_species_type_validation(self):
        invalid_options = [True, 0.1, 0, {1, 2}, (1, 2), [1, 2]]
        for species in invalid_options:
            with pytest.raises(ValueError, match="species must be a string or None"):
                SubjectDescription(id="subject_1", species=species)  # type: ignore

        with pytest.raises(ValueError, match="species cannot be an empty string"):
            SubjectDescription(id="subject_1", species="")

    def test_zero_age(self):
        result = SubjectDescription(id="subject_1", species=None, age=0)
        assert result.age == 0.0

    def test_negative_age(self):
        with pytest.raises(ValueError, match="age cannot be negative"):
            SubjectDescription(id="subject_1", age=-5)

    def test_negative_age_float(self):
        with pytest.raises(ValueError, match="age cannot be negative"):
            SubjectDescription(id="subject_1", age=-5.5)

    def test_negative_age_string(self):
        with pytest.raises(ValueError, match="age cannot be negative"):
            SubjectDescription(id="subject_1", age="-10")

    def test_age_with_unexpected_type(self):
        with pytest.raises(
            Exception, match="age must be a float, int, numeric string, or None"
        ):
            SubjectDescription(id="subject_1", age=[])  # type: ignore
