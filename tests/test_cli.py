"""Tests for CLI commands in brainsets._cli module."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from brainsets._cli.cli import cli


class TestPrepareCommand:
    """Tests for the 'brainsets prepare' command."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        raw_dir.mkdir()
        processed_dir.mkdir()
        return {
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
        }

    def test_prepare_valid_brainset(self, mock_config):
        """Test prepare command with valid brainset constructs correct subprocess call.
        Ensuring it passes through the inline metadata correctly.
        """
        runner = CliRunner()

        with (
            patch("brainsets._cli.cli_prepare.load_config", return_value=mock_config),
            patch("brainsets._cli.cli_prepare.subprocess.run") as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(cli, ["prepare", "pei_pandarinath_nlb_2021"])
            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert "Preparing pei_pandarinath_nlb_2021" in result.output

            # Verify subprocess was called with correct arguments
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            command = call_args[1].get("command") or call_args[0][0]
            env = call_args[1].get("env")

            assert command[0] == "uv"
            assert command[1] == "run"
            assert command[2] == "--directory"
            assert "pei_pandarinath_nlb_2021" in command[3]
            assert "brainsets_pipelines" in command[3]
            assert command[4] == "--isolated"
            assert command[5] == "--no-project"
            assert "--python" in command
            assert command[command.index("--python") + 1] == "3.11"

            if "--with-editable" in command:
                editable_spec = command[command.index("--with-editable") + 1]
                assert ("brainsets" in editable_spec) and ("file://" in editable_spec)

            assert "--with" in command
            with_spec = command[command.index("--with") + 1]
            assert "dandi==0.71.3" in with_spec
            assert ("brainsets" in with_spec) or ("--with-editable" in command)

            assert "python" in command
            assert "-m" in command
            assert "brainsets.runner" in command
            assert any("pipeline.py" in arg for arg in command)
            assert f"--raw-dir={mock_config['raw_dir']}" in command
            assert f"--processed-dir={mock_config['processed_dir']}" in command
            assert "-c4" in command  # default cores
            assert env is not None
            assert env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"

    def test_cli_raw_processed_dirs_override(self, tmp_path):
        """Test prepare command with raw and processed dirs overridden."""
        runner = CliRunner()

        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        raw_dir.mkdir()
        processed_dir.mkdir()

        with (patch("brainsets._cli.cli_prepare.subprocess.run") as mock_subprocess,):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(
                cli,
                [
                    "prepare",
                    "pei_pandarinath_nlb_2021",
                    "--raw-dir",
                    str(raw_dir),
                    "--processed-dir",
                    str(processed_dir),
                ],
            )
            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert "Preparing pei_pandarinath_nlb_2021" in result.output

            # Verify subprocess was called with correct arguments
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            command = call_args[1].get("command") or call_args[0][0]

            assert f"--raw-dir={str(raw_dir)}" in command
            assert f"--processed-dir={str(processed_dir)}" in command

    def test_extra_option_passthrough(self, mock_config):
        """Test that extra options are passed through to the subprocess."""
        runner = CliRunner()

        with (
            patch("brainsets._cli.cli_prepare.load_config", return_value=mock_config),
            patch("brainsets._cli.cli_prepare.subprocess.run") as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(
                cli, ["prepare", "pei_pandarinath_nlb_2021", "--unknown", "option"]
            )
            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert "Preparing pei_pandarinath_nlb_2021" in result.output

            # Verify subprocess was called with correct arguments
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            command = call_args[1].get("command") or call_args[0][0]

            assert "--unknown" in command[-2]
            assert "option" in command[-1]

    def test_prepare_respects_user_ray_uv_setting(self, mock_config):
        """Test prepare command does not override user's Ray uv runtime setting."""
        runner = CliRunner()

        with (
            patch.dict(os.environ, {"RAY_ENABLE_UV_RUN_RUNTIME_ENV": "1"}, clear=False),
            patch("brainsets._cli.cli_prepare.load_config", return_value=mock_config),
            patch("brainsets._cli.cli_prepare.subprocess.run") as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(cli, ["prepare", "pei_pandarinath_nlb_2021"])
            assert result.exit_code == 0, f"CLI failed with: {result.output}"

            call_args = mock_subprocess.call_args
            assert call_args[1].get("env") is None
