"""Tests for bridge CLI using click.testing.CliRunner."""

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from bridge.cli import cli
from bridge.secrets import Secrets, write_secrets


class TestInitCommand:
    """Tests for `claude-discord-bridge init` subcommand."""

    def test_init_writes_secrets_file(self, tmp_path: Path, monkeypatch) -> None:
        """init with simulated stdin writes a secrets file at the expected location."""
        secrets_file = tmp_path / "secrets.json"
        monkeypatch.setenv("BRIDGE_SECRETS_PATH", str(secrets_file))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["init"], input="test_token_abc\n12345\n"
        )

        assert result.exit_code == 0
        assert secrets_file.exists()

    def test_init_sets_0600_perms(self, tmp_path: Path, monkeypatch) -> None:
        """init writes a secrets file with exactly 0o600 permissions."""
        import stat

        secrets_file = tmp_path / "secrets.json"
        monkeypatch.setenv("BRIDGE_SECRETS_PATH", str(secrets_file))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["init"], input="test_token_abc\n12345\n"
        )

        assert result.exit_code == 0
        perms = stat.S_IMODE(secrets_file.stat().st_mode)
        assert perms == 0o600

    def test_init_rejects_non_integer_channel_id(self, tmp_path: Path, monkeypatch) -> None:
        """init rejects a non-integer channel ID by reprompting."""
        secrets_file = tmp_path / "secrets.json"
        monkeypatch.setenv("BRIDGE_SECRETS_PATH", str(secrets_file))

        runner = CliRunner()
        # Input: first bad channel ID, then a good one
        result = runner.invoke(
            cli, ["init"], input="test_token_abc\nnot_a_number\n12345\n"
        )

        assert result.exit_code == 0
        assert secrets_file.exists()

    def test_init_aborts_if_file_exists_and_user_says_no(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """init aborts cleanly when secrets file already exists and user answers no."""
        secrets_file = tmp_path / "secrets.json"
        monkeypatch.setenv("BRIDGE_SECRETS_PATH", str(secrets_file))

        # Pre-create the secrets file
        write_secrets(Secrets(bot_token="old_token", channel_id=999), path=secrets_file)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["init"], input="n\n"
        )

        # Should abort with exit code 1 (via abort())
        assert result.exit_code == 1

    def test_init_prompts_for_token_and_channel(self, tmp_path: Path, monkeypatch) -> None:
        """init prompts interactively for bot token and channel ID."""
        secrets_file = tmp_path / "secrets.json"
        monkeypatch.setenv("BRIDGE_SECRETS_PATH", str(secrets_file))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["init"], input="my_token\n54321\n"
        )

        assert result.exit_code == 0
        assert "DISCORD_BOT_TOKEN" in result.output or "token" in result.output.lower()
        assert "DISCORD_CHANNEL_ID" in result.output or "channel" in result.output.lower()

    def test_init_prints_success_message(self, tmp_path: Path, monkeypatch) -> None:
        """init prints a success message mentioning secrets.json and 0600."""
        secrets_file = tmp_path / "secrets.json"
        monkeypatch.setenv("BRIDGE_SECRETS_PATH", str(secrets_file))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["init"], input="test_token\n12345\n"
        )

        assert result.exit_code == 0
        assert "secrets.json" in result.output or "wrote" in result.output
        assert "0600" in result.output


class TestServeCommand:
    """Tests for `claude-discord-bridge serve` subcommand."""

    def test_serve_help_prints_help_text(self) -> None:
        """serve --help prints the help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])

        assert result.exit_code == 0
        assert "serve" in result.output.lower() or "Usage:" in result.output

    def test_serve_without_secrets_exits_2(self, tmp_path: Path, monkeypatch) -> None:
        """serve with no secrets file exits 2 and prints a clear error."""
        secrets_file = tmp_path / "nonexistent.json"
        monkeypatch.setenv("BRIDGE_SECRETS_PATH", str(secrets_file))

        runner = CliRunner()
        result = runner.invoke(cli, ["serve"])

        assert result.exit_code == 2
        assert "init" in result.output.lower() or "not found" in result.output.lower()

    def test_serve_help_includes_host_port_options(self) -> None:
        """serve --help includes --host and --port options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])

        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
