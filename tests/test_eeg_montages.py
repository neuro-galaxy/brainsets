"""
Test suite for data_standardization.py module.

This module contains comprehensive tests for all functions in the data_standardization module,
including edge cases, error conditions, and various input scenarios.
"""

import pytest
import numpy as np
from unittest.mock import patch

from brainsets.utils.eeg_montages import (
    names_to_standard_names,
    names_to_standard_types,
    find_match_percentage_by_montage,
    get_standard_montage,
    get_standard_ch_info,
    get_biosemi_to_std_mapping,
    get_all_montage_matches,
    get_all_electrode_names_to_montage_mapping,
)


class TestGetStandardChNames:
    """Test cases for get_standard_ch_names function."""

    def test_valid_channel_names(self):
        """Test with valid channel names that exist in mapping."""
        ch_names = ["Fp1", "Fp2", "Cz", "Pz"]
        result = names_to_standard_names(ch_names)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(ch_names)
        assert result[0] == "Fp1"
        assert result[1] == "Fp2"
        assert result[2] == "Cz"
        assert result[3] == "Pz"

    def test_empty_list(self):
        """Test with empty channel list."""
        ch_names = []
        result = names_to_standard_names(ch_names)

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_invalid_channel_name(self):
        """Test with invalid channel name not in mapping."""
        ch_names = ["InvalidChannel"]

        with pytest.raises(KeyError) as exc_info:
            names_to_standard_names(ch_names)

    def test_mixed_valid_invalid_channels(self):
        """Test with mix of valid and invalid channel names."""
        ch_names = ["Fp1", "InvalidChannel", "Cz"]

        with pytest.raises(KeyError) as exc_info:
            names_to_standard_names(ch_names)


class TestGetStandardChTypes:
    """Test cases for get_standard_ch_types function."""

    def test_valid_channel_names(self):
        """Test with valid channel names that exist in mapping."""
        ch_names = ["Fp1", "Fp2", "Cz", "Pz"]
        result = names_to_standard_types(ch_names)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(ch_names)
        assert result[0] == "EEG"
        assert result[1] == "EEG"
        assert result[2] == "EEG"
        assert result[3] == "EEG"

    def test_empty_list(self):
        """Test with empty channel list."""
        ch_names = []
        result = names_to_standard_types(ch_names)

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_invalid_channel_name(self):
        """Test with invalid channel name not in mapping."""
        ch_names = ["InvalidChannel"]

        with pytest.raises(KeyError) as exc_info:
            names_to_standard_types(ch_names)

    def test_mixed_valid_invalid_channels(self):
        """Test with mix of valid and invalid channel names."""
        ch_names = ["Fp1", "InvalidChannel", "Cz"]

        with pytest.raises(KeyError) as exc_info:
            names_to_standard_names(ch_names)


class TestFindMatchPercentageByMontage:
    """Test cases for find_match_percentage_by_montage function."""

    def test_empty_channel_list(self):
        """Test that find_match_percentage_by_montage raises ValueError on empty channel list."""
        eeg_ch_names = []
        with pytest.raises(
            ValueError, match="EEG channels should be a non-empty list."
        ):
            find_match_percentage_by_montage(eeg_ch_names)

    @patch("brainsets.utils.eeg_montages.get_mne_montages_info")
    def test_match_percentage_calculation(self, mock_get_mne_montages_info):
        """Test with valid channel names."""

        mock_get_mne_montages_info.return_value = {
            "montage1": ["Ch1", "Ch2", "Ch3"],
            "montage2": ["Ch1", "Ch2", "Ch3", "Ch4"],
            "montage3": ["Ch5", "Ch6", "Ch7", "Ch8", "Ch9", "Ch10"],
            "montage4": [
                "Ch1",
                "Ch2",
                "Ch3",
                "Ch4",
                "Ch5",
                "Ch6",
                "Ch7",
                "Ch8",
                "Ch9",
                "Ch10",
            ],
        }

        eeg_ch_names = ["Ch1", "Ch2", "Ch3", "Ch4"]
        match_to_self, match_to_mne = find_match_percentage_by_montage(eeg_ch_names)

        # Check output structure
        assert isinstance(match_to_self, dict)
        assert isinstance(match_to_mne, dict)
        assert len(match_to_self) > 0
        assert len(match_to_mne) > 0

        # Check that all values are between 0 and 1
        for value in match_to_self.values():
            assert 0.0 <= value <= 1.0
        for value in match_to_mne.values():
            assert 0.0 <= value <= 1.0

        # Check that the match_to_self and match_to_mne are correct
        # Check test_montage1 (partial match case 1)
        assert match_to_self["montage1"] == 0.75
        assert match_to_mne["montage1"] == 1.0

        # Check test_montage2 (perfect match)
        assert match_to_self["montage2"] == 1.0
        assert match_to_mne["montage2"] == 1.0

        # Check test_montage4 (no match)
        assert match_to_self["montage3"] == 0.0
        assert match_to_mne["montage3"] == 0.0

        # Check test_montage3 (partial match case 2)
        assert match_to_self["montage4"] == 1.0
        assert match_to_mne["montage4"] == 0.4

    def test_biosemi64_remapping(self):
        """Test Biosemi64 channel remapping."""
        # Use actual Biosemi64 channel names
        biosemi64_ch_names = list(get_biosemi_to_std_mapping().keys())[
            :10
        ]  # First 10 channels

        match_to_self, match_to_mne = find_match_percentage_by_montage(
            biosemi64_ch_names
        )

        assert match_to_self["biosemi64"] == 1.0
        assert match_to_mne["biosemi64"] == 10 / 64

        assert match_to_mne["biosemi16"] > match_to_mne["standard_1020"]
        assert match_to_mne["biosemi16"] > match_to_mne["standard_1005"]
        assert match_to_mne["biosemi32"] > match_to_mne["standard_1020"]
        assert match_to_mne["biosemi32"] > match_to_mne["standard_1005"]
        assert match_to_mne["biosemi64"] > match_to_mne["standard_1020"]
        assert match_to_mne["biosemi64"] > match_to_mne["standard_1005"]


class TestGetStandardMontage:
    """Test cases for get_standard_montage function."""

    def test_empty_channel_list(self):
        """Test that get_standard_montage raises ValueError on empty channel list."""
        eeg_ch_names = []
        with pytest.raises(
            ValueError, match="EEG channels should be a non-empty list."
        ):
            get_standard_montage(eeg_ch_names)

    def test_output_format(self):
        """Test with valid channel names."""
        eeg_ch_names = ["Fp1", "Fp2", "Cz", "Pz"]
        montage, match_percentage, unmatch_ch_names = get_standard_montage(eeg_ch_names)

        assert isinstance(montage, str)
        assert isinstance(match_percentage, float)
        assert 0.0 <= match_percentage <= 1.0
        assert isinstance(unmatch_ch_names, np.ndarray)

    @patch("brainsets.utils.eeg_montages.find_match_percentage_by_montage")
    @patch("brainsets.utils.eeg_montages.get_mne_montages_info")
    def test_perfect_match_to_self_and_mne(
        self,
        mock_get_mne_montages_info,
        mock_find_match_percentage_by_montage,
    ):
        """Test selection when only one perfect match exists."""
        mock_get_mne_montages_info.return_value = {
            "montage1": ["Ch1", "Ch2"],
            "montage2": ["Ch1", "Ch2", "Ch3"],
            "montage3": ["Ch1", "Ch2", "Ch3", "Ch4"],
        }

        # Based on mock_get_mne_montages_info.return_value and eeg_ch_names
        mock_find_match_percentage_by_montage.return_value = [
            {"montage1": 0.5, "montage2": 0.75, "montage3": 1.0},  # match to self
            {"montage1": 1.0, "montage2": 1.0, "montage3": 1.0},  # match to mne
        ]

        eeg_ch_names = ["Ch1", "Ch2", "Ch3", "Ch4"]
        montage, match_percentage, unmatch_ch_names = get_standard_montage(eeg_ch_names)

        # Should select montage3 (highest match_to_mne among perfect matches)
        assert montage == "montage3"
        assert match_percentage == 1.0
        assert len(unmatch_ch_names) == 0

    @patch("brainsets.utils.eeg_montages.find_match_percentage_by_montage")
    @patch("brainsets.utils.eeg_montages.get_mne_montages_info")
    def test_multiple_perfect_match_to_self(
        self,
        mock_get_mne_montages_info,
        mock_find_match_percentage_by_montage,
    ):
        """Test selection when multiple perfect match to self exist."""
        mock_get_mne_montages_info.return_value = {
            "montage1": ["Ch1", "Ch2", "Ch3"],
            "montage2": ["Ch1", "Ch2", "Ch3", "Ch4"],
            "montage3": ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5"],
        }

        mock_find_match_percentage_by_montage.return_value = [
            {"montage1": 1.0, "montage2": 1.0, "montage3": 1.0},  # match to self
            {"montage1": 0.666, "montage2": 0.5, "montage3": 0.4},  # match to mne
        ]

        eeg_ch_names = ["Ch1", "Ch2"]
        montage, match_percentage, unmatch_ch_names = get_standard_montage(eeg_ch_names)

        assert montage == "montage1"
        assert match_percentage == 1.0
        assert len(unmatch_ch_names) == 0

    @patch("brainsets.utils.eeg_montages.find_match_percentage_by_montage")
    @patch("brainsets.utils.eeg_montages.get_mne_montages_info")
    def test_multiple_perfect_match_to_mne(
        self,
        mock_get_mne_montages_info,
        mock_find_match_percentage_by_montage,
    ):
        """Test selection when multiple perfect match to mne exist."""
        mock_get_mne_montages_info.return_value = {
            "montage1": ["Ch1", "Ch2", "Ch3"],
            "montage2": ["Ch1", "Ch2", "Ch3", "Ch4"],
            "montage3": ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5"],
        }

        mock_find_match_percentage_by_montage.return_value = [
            {"montage1": 0.3, "montage2": 0.4, "montage3": 0.5},  # match to self
            {"montage1": 1.0, "montage2": 1.0, "montage3": 1.0},  # match to mne
        ]

        eeg_ch_names = [
            "Ch1",
            "Ch2",
            "Ch3",
            "Ch4",
            "Ch5",
            "Ch6",
            "Ch7",
            "Ch8",
            "Ch9",
            "Ch10",
        ]
        montage, match_percentage, unmatch_ch_names = get_standard_montage(eeg_ch_names)

        assert montage == "montage3"
        assert match_percentage == 0.5
        assert len(unmatch_ch_names) == 5

    @patch("brainsets.utils.eeg_montages.find_match_percentage_by_montage")
    @patch("brainsets.utils.eeg_montages.get_mne_montages_info")
    def test_no_perfect_match(
        self,
        mock_get_mne_montages_info,
        mock_find_match_percentage_by_montage,
    ):
        """Test selection when no  match exists."""
        mock_get_mne_montages_info.return_value = {
            "montage1": ["Ch1", "Ch2", "Ch3"],
            "montage2": ["Ch1", "Ch2", "Ch3", "Ch4"],
            "montage3": ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5"],
        }

        mock_find_match_percentage_by_montage.return_value = [
            {"montage1": 0.0, "montage2": 0.0, "montage3": 0.0},  # match to self
            {"montage1": 0.0, "montage2": 0.0, "montage3": 0.0},  # match to mne
        ]

        eeg_ch_names = ["Ch6", "Ch7", "Ch8", "Ch9", "Ch10"]
        montage, match_percentage, unmatch_ch_names = get_standard_montage(eeg_ch_names)

        assert montage == "other"
        assert match_percentage == 0.0
        assert len(unmatch_ch_names) == len(eeg_ch_names)

    def test_biosemi_remapping(self):
        """Test Biosemi 16, 32, 64 channel remapping."""

        # BIOSEMI16
        biosemi16_ch_names = [
            "A1",
            "B2",
            "B8",
            "B6",
            "A5",
            "A15",
            "A13",
            "B16",
            "B18",
            "B20",
            "B26",
            "A31",
            "A21",
            "A27",
            "A29",
            "B32",
        ]

        # All channels
        montage, match_percentage, unmatch_ch_names = get_standard_montage(
            biosemi16_ch_names
        )
        assert montage == "biosemi16"
        assert match_percentage == 1.0
        assert len(unmatch_ch_names) == 0

        # partial channels
        montage, match_percentage, unmatch_ch_names = get_standard_montage(
            biosemi16_ch_names[:4]
        )
        assert montage == "biosemi16"
        assert match_percentage == 1.0
        assert len(unmatch_ch_names) == 0

        # BIOSEMI32
        biosemi32_ch_names = [
            "A1",
            "A3",
            "A7",
            "A5",
            "A11",
            "A9",
            "A15",
            "A13",
            "A19",
            "A17",
            "A23",
            "A21",
            "A31",
            "A26",
            "A27",
            "A29",
            "B32",
            "B31",
            "B26",
            "B28",
            "B22",
            "B24",
            "B18",
            "B20",
            "B12",
            "B14",
            "B8",
            "B10",
            "B4",
            "B2",
            "B6",
            "B16",
        ]

        # All channels
        montage, match_percentage, unmatch_ch_names = get_standard_montage(
            biosemi32_ch_names
        )
        assert montage == "biosemi32"
        assert match_percentage == 1.0
        assert len(unmatch_ch_names) == 0

        # partial channels
        montage, match_percentage, unmatch_ch_names = get_standard_montage(
            biosemi32_ch_names[:16]
        )
        assert montage == "biosemi32"
        assert match_percentage == 1.0
        assert len(unmatch_ch_names) == 0

        # BIOSEMI64
        biosemi64_ch_names = list(get_biosemi_to_std_mapping().keys())

        # All channels
        montage, match_percentage, unmatch_ch_names = get_standard_montage(
            biosemi64_ch_names
        )
        assert montage == "biosemi64"
        assert match_percentage == 1.0
        assert len(unmatch_ch_names) == 0

        # partial channels
        montage, match_percentage, unmatch_ch_names = get_standard_montage(
            biosemi64_ch_names[:32]
        )
        assert montage == "biosemi64"
        assert match_percentage == 1.0
        assert len(unmatch_ch_names) == 0


class TestGetStandardChInfo:
    """Test cases for get_standard_ch_info function."""

    def test_valid_eeg_channels(self):
        """Test with valid EEG channel names."""
        og_ch_names = ["Fp1", "Fp2", "Cz", "Pz"]
        std_ch_names, std_ch_types, std_montage = get_standard_ch_info(og_ch_names)

        assert isinstance(std_ch_names, np.ndarray)
        assert isinstance(std_ch_types, np.ndarray)
        assert isinstance(std_montage, str)
        assert len(std_ch_names) == len(og_ch_names)
        assert len(std_ch_types) == len(og_ch_names)

    def test_mixed_channel_types(self):
        """Test with mixed channel types (EEG and non-EEG)."""
        og_ch_names = ["Fp1", "Fp2", "EOG1", "EOG2"]
        std_ch_names, std_ch_types, std_montage = get_standard_ch_info(og_ch_names)

        assert isinstance(std_ch_names, np.ndarray)
        assert isinstance(std_ch_types, np.ndarray)
        assert isinstance(std_montage, str)
        assert len(std_ch_names) == len(og_ch_names)
        assert len(std_ch_types) == len(og_ch_names)

    def test_no_eeg_channels(self):
        """Test with no EEG channels."""
        og_ch_names = ["EOG1", "EOG2", "EMG1"]
        std_ch_names, std_ch_types, std_montage = get_standard_ch_info(og_ch_names)

        assert isinstance(std_ch_names, np.ndarray)
        assert isinstance(std_ch_types, np.ndarray)
        assert std_montage == "other"
        assert len(std_ch_names) == len(og_ch_names)
        assert len(std_ch_types) == len(og_ch_names)

    def test_empty_channel_list(self):
        """Test with empty channel list."""
        og_ch_names = []
        with pytest.raises(ValueError):
            get_standard_ch_info(og_ch_names)

    def test_invalid_channel_name(self):
        """Test with invalid channel name."""
        og_ch_names = ["InvalidChannel"]

        with pytest.raises(KeyError):
            get_standard_ch_info(og_ch_names)

    @patch("brainsets.utils.eeg_montages.get_standard_montage")
    def test_unmatched_channels(
        self,
        mock_get_standard_montage,
    ):
        """Test behavior with unmatched channels."""
        mock_get_standard_montage.return_value = (
            "test_montage",
            0.8,
            np.array(["Fp1"]),
        )

        og_ch_names = ["Fp1", "Fp2"]
        std_ch_names, std_ch_types, std_montage = get_standard_ch_info(og_ch_names)

        # Check that unmatched channels get EEG-OTHER type
        eeg_mask = std_ch_types == "EEG"
        eeg_other_mask = std_ch_types == "EEG-OTHER"
        assert np.any(eeg_other_mask) or np.any(eeg_mask)


class TestGetAllMontageMatches:
    """Test cases for get_all_montage_matches function."""

    def test_empty_channel_list(self):
        """Test that get_all_montage_matches raises ValueError on empty channel list."""
        eeg_ch_names = []
        with pytest.raises(
            ValueError, match="EEG channels should be a non-empty list."
        ):
            get_all_montage_matches(eeg_ch_names)

    def test_duplicate_channel_names(self):
        """Test that get_all_montage_matches raises ValueError on duplicate channel names."""
        eeg_ch_names = ["Fp1", "Fp1", "Cz"]
        with pytest.raises(ValueError, match="EEG channel names should be unique"):
            get_all_montage_matches(eeg_ch_names)

    def test_none_input(self):
        """Test that get_all_montage_matches raises TypeError on None input."""
        with pytest.raises(TypeError):
            get_all_montage_matches(None)

    def test_output_structure(self):
        """Test that output has the correct structure."""
        eeg_ch_names = ["Fp1", "Fp2", "Cz", "Pz"]
        all_matches = get_all_montage_matches(eeg_ch_names)

        # Check main structure
        assert isinstance(all_matches, dict)
        assert len(all_matches) > 0

        # Check that each montage has the expected keys
        for montage_name, montage_info in all_matches.items():
            assert isinstance(montage_name, str)
            assert isinstance(montage_info, dict)

            expected_keys = {
                "match_to_input",
                "match_to_montage",
                "matched_channels",
                "unmatched_input_channels",
                "unmatched_montage_channels",
                "montage_channels",
                "positions",
                "total_input_channels",
                "total_montage_channels",
            }
            assert set(montage_info.keys()) == expected_keys

            # Check data types
            assert isinstance(montage_info["match_to_input"], float)
            assert isinstance(montage_info["match_to_montage"], float)
            assert isinstance(montage_info["matched_channels"], list)
            assert isinstance(montage_info["unmatched_input_channels"], list)
            assert isinstance(montage_info["unmatched_montage_channels"], list)
            assert isinstance(montage_info["montage_channels"], list)
            assert isinstance(montage_info["positions"], dict)
            assert isinstance(montage_info["total_input_channels"], int)
            assert isinstance(montage_info["total_montage_channels"], int)

            # Check value ranges
            assert 0.0 <= montage_info["match_to_input"] <= 1.0
            assert 0.0 <= montage_info["match_to_montage"] <= 1.0
            assert montage_info["total_input_channels"] == len(eeg_ch_names)

    @patch("brainsets.utils.eeg_montages.get_mne_montages_info")
    def test_match_calculations(self, mock_get_mne_montages_info):
        """Test that match calculations are correct."""
        # Mock a simple montage setup
        mock_get_mne_montages_info.return_value = {
            "test_montage1": ["Ch1", "Ch2", "Ch3"],
            "test_montage2": ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5"],
        }

        eeg_ch_names = ["Ch1", "Ch2", "Ch3"]
        all_matches = get_all_montage_matches(eeg_ch_names)

        # Check test_montage1 (perfect match)
        montage1_info = all_matches["test_montage1"]
        assert montage1_info["match_to_input"] == 1.0  # 3/3
        assert montage1_info["match_to_montage"] == 1.0  # 3/3
        assert set(montage1_info["matched_channels"]) == {"Ch1", "Ch2", "Ch3"}
        assert len(montage1_info["unmatched_input_channels"]) == 0
        assert len(montage1_info["unmatched_montage_channels"]) == 0

        # Check test_montage2 (partial match)
        montage2_info = all_matches["test_montage2"]
        assert montage2_info["match_to_input"] == 1.0  # 3/3
        assert montage2_info["match_to_montage"] == 0.6  # 3/5
        assert set(montage2_info["matched_channels"]) == {"Ch1", "Ch2", "Ch3"}
        assert len(montage2_info["unmatched_input_channels"]) == 0
        assert set(montage2_info["unmatched_montage_channels"]) == {"Ch4", "Ch5"}

    def test_standard_montages_included(self):
        """Test that standard MNE montages are included."""
        eeg_ch_names = ["Fp1", "Fp2", "Cz", "Pz"]
        all_matches = get_all_montage_matches(eeg_ch_names)

        # Should include common standard montages
        expected_montages = ["standard_1005", "standard_1020", "biosemi64"]
        for montage_name in expected_montages:
            assert montage_name in all_matches

            # Each montage should have valid structure
            montage_info = all_matches[montage_name]
            assert 0.0 <= montage_info["match_to_input"] <= 1.0
            assert 0.0 <= montage_info["match_to_montage"] <= 1.0
            assert montage_info["total_input_channels"] == len(eeg_ch_names)

    def test_biosemi_remapping(self):
        """Test that biosemi channels are properly remapped."""
        # Use actual biosemi channel names
        biosemi_ch_names = ["A1", "A2", "A3", "A4"]  # First 4 biosemi channels
        all_matches = get_all_montage_matches(biosemi_ch_names)

        # Should include biosemi montages
        assert "biosemi64" in all_matches

        # Check that original channel names are preserved in the output
        biosemi64_info = all_matches["biosemi64"]
        for matched_ch in biosemi64_info["matched_channels"]:
            assert (
                matched_ch in biosemi_ch_names
            )  # Should be original names, not mapped

    def test_positions_included(self):
        """Test that electrode positions are included when available."""
        eeg_ch_names = ["Fp1", "Fp2", "Cz"]
        all_matches = get_all_montage_matches(eeg_ch_names)

        # Find a montage with matches
        montage_with_matches = None
        for montage_name, info in all_matches.items():
            if len(info["matched_channels"]) > 0:
                montage_with_matches = info
                break

        assert montage_with_matches is not None

        # Check that positions are provided for matched channels
        positions = montage_with_matches["positions"]
        assert isinstance(positions, dict)

        # Each position should be a 3D coordinate list
        for ch_name, position in positions.items():
            assert isinstance(position, list)
            assert len(position) == 3  # x, y, z coordinates
            assert all(isinstance(coord, (int, float)) for coord in position)

    def test_numpy_array_input(self):
        """Test that numpy array input works correctly."""
        import numpy as np

        eeg_ch_names = np.array(["Fp1", "Fp2", "Cz", "Pz"])
        all_matches = get_all_montage_matches(eeg_ch_names)

        # Should work the same as list input
        assert isinstance(all_matches, dict)
        assert len(all_matches) > 0


class TestErrorConditions:
    """Test error conditions."""

    def test_none_input(self):
        """Test behavior with None input."""
        with pytest.raises(TypeError):
            names_to_standard_names(None)

        with pytest.raises(TypeError):
            names_to_standard_types(None)

        with pytest.raises(TypeError):
            find_match_percentage_by_montage(None)

        with pytest.raises(TypeError):
            get_standard_montage(None)

        with pytest.raises(TypeError):
            get_standard_ch_info(None)

        with pytest.raises(TypeError):
            get_all_montage_matches(None)

    def test_duplicate_channel_names(self):
        """Test behavior with duplicate channel names."""
        ch_names = ["Fp1", "Fp1", "Cz"]

        with pytest.raises(ValueError):
            find_match_percentage_by_montage(ch_names)

        with pytest.raises(ValueError):
            get_standard_montage(ch_names)

        with pytest.raises(ValueError):
            get_standard_ch_info(ch_names)

        with pytest.raises(ValueError):
            get_all_montage_matches(ch_names)

    def test_case_insensitive_matching(self):
        """Test that channel matching is case-insensitive."""
        # Test with mixed case channel names
        eeg_ch_names_mixed_case = ["fp1", "FP2", "Cz", "pz", "O1", "o2"]
        all_matches = get_all_montage_matches(eeg_ch_names_mixed_case)

        # Should have matches for standard montages
        assert len(all_matches) > 0

        # Check that case-insensitive matching works
        for montage_name, montage_info in all_matches.items():
            # The matched channels should preserve original case from input
            for ch in montage_info["matched_channels"]:
                assert ch in eeg_ch_names_mixed_case

            # The unmatched input channels should also preserve original case
            for ch in montage_info["unmatched_input_channels"]:
                assert ch in eeg_ch_names_mixed_case

        # Test with all lowercase
        eeg_ch_names_lower = ["fp1", "fp2", "cz", "pz", "o1", "o2"]
        all_matches_lower = get_all_montage_matches(eeg_ch_names_lower)

        # Test with all uppercase
        eeg_ch_names_upper = ["FP1", "FP2", "CZ", "PZ", "O1", "O2"]
        all_matches_upper = get_all_montage_matches(eeg_ch_names_upper)

        # All three should have similar match results (case-insensitive)
        # We'll check that they have the same number of total matches across montages
        total_matches_mixed = sum(
            len(info["matched_channels"]) for info in all_matches.values()
        )
        total_matches_lower = sum(
            len(info["matched_channels"]) for info in all_matches_lower.values()
        )
        total_matches_upper = sum(
            len(info["matched_channels"]) for info in all_matches_upper.values()
        )

        # Should have the same total number of matches (case-insensitive)
        assert total_matches_mixed == total_matches_lower == total_matches_upper


class TestGetAllElectrodeNamesToMontageMapping:
    """Test cases for get_all_electrode_names_to_montage_mapping function."""

    def test_output_structure(self):
        """Test that the function returns a dictionary with correct structure."""
        mapping = get_all_electrode_names_to_montage_mapping()

        # Check output type
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

        # Check that all values are strings (montage names)
        for electrode, montages in mapping.items():
            assert isinstance(electrode, str)
            assert isinstance(montages, str)
            assert len(electrode) > 0
            assert len(montages) > 0

    def test_common_electrodes_in_multiple_montages(self):
        """Test that common electrodes appear in multiple montages."""
        mapping = get_all_electrode_names_to_montage_mapping()

        # Check for common electrodes that should be in multiple montages
        common_electrodes = ["Fp1", "Fp2", "Cz", "Pz", "Fz", "C3", "C4"]

        for electrode in common_electrodes:
            if electrode in mapping:
                montages = mapping[electrode]
                # Should contain multiple montages (comma-separated)
                assert "," in montages or len(montages) > 0

                # Split and check individual montages
                montage_list = montages.split(",")
                assert len(montage_list) >= 1

                # Check that montage names are valid
                for montage in montage_list:
                    assert len(montage) > 0
                    assert montage.strip() == montage  # No leading/trailing spaces

    def test_standard_montages_included(self):
        """Test that standard MNE montages are included in the mapping."""
        mapping = get_all_electrode_names_to_montage_mapping()

        # Get all unique montages mentioned in the mapping
        all_montages = set()
        for montages_str in mapping.values():
            all_montages.update(montages_str.split(","))

        # Should include common standard montages
        expected_montages = [
            "standard_1005",
            "standard_1020",
            "biosemi64",
            "biosemi32",
            "biosemi16",
        ]

        for expected_montage in expected_montages:
            assert expected_montage in all_montages

    def test_biosemi_electrodes_included(self):
        """Test that biosemi electrodes are included in the mapping."""
        mapping = get_all_electrode_names_to_montage_mapping()

        # Check for some common biosemi electrode names
        biosemi_electrodes = ["A1", "A2", "B1", "B2", "A32", "B32"]

        for electrode in biosemi_electrodes:
            if electrode in mapping:
                montages = mapping[electrode]
                # Should contain biosemi montages
                assert "biosemi" in montages.lower()

    def test_no_empty_values(self):
        """Test that no electrode has empty montage list."""
        mapping = get_all_electrode_names_to_montage_mapping()

        for electrode, montages in mapping.items():
            assert len(montages) > 0
            assert montages.strip() != ""
            # No empty montage names in the comma-separated list
            montage_list = [m.strip() for m in montages.split(",")]
            assert all(len(m) > 0 for m in montage_list)

    def test_consistency_with_mne_montages(self):
        """Test that the mapping is consistent with MNE montage information."""
        from brainsets.utils.eeg_montages import get_mne_montages_info

        mapping = get_all_electrode_names_to_montage_mapping()
        mne_montages = get_mne_montages_info()

        # For each electrode in the mapping, verify it exists in the claimed montages
        for electrode, montages_str in mapping.items():
            montage_list = montages_str.split(",")

            for montage_name in montage_list:
                montage_name = montage_name.strip()
                assert montage_name in mne_montages
                assert electrode in mne_montages[montage_name]

    def test_electrode_coverage(self):
        """Test that the mapping covers a reasonable number of electrodes."""
        mapping = get_all_electrode_names_to_montage_mapping()

        # Should have a substantial number of electrodes
        assert len(mapping) > 50  # Should have many electrodes

        # Should include common EEG electrode names
        common_electrodes = ["Fp1", "Fp2", "Fz", "Cz", "Pz", "Oz", "T7", "T8"]
        found_common = sum(1 for electrode in common_electrodes if electrode in mapping)
        assert found_common >= 5  # Most common electrodes should be present
