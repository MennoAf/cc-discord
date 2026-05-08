"""Tests for transcript JSONL reading and assistant-turn extraction."""

from pathlib import Path
import json

from bridge.transcript import read_entries, extract_final_assistant_text


class TestReadEntries:
    """Tests for the read_entries generator."""

    def test_yields_valid_json_lines(self, tmp_path: Path) -> None:
        """read_entries yields each valid JSON line as a dict."""
        transcript = tmp_path / "transcript.jsonl"
        entries = [
            {"type": "user", "message": {"role": "user", "content": "hi"}},
            {"type": "assistant", "message": {"role": "assistant", "content": []}},
        ]
        with open(transcript, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = list(read_entries(transcript))
        assert len(result) == 2
        assert result[0]["type"] == "user"
        assert result[1]["type"] == "assistant"

    def test_skips_malformed_json_lines(self, tmp_path: Path) -> None:
        """read_entries silently skips lines with invalid JSON."""
        transcript = tmp_path / "transcript.jsonl"
        with open(transcript, "w") as f:
            f.write('{"type": "user"}\n')
            f.write('not valid json\n')
            f.write('{"type": "assistant"}\n')

        result = list(read_entries(transcript))
        assert len(result) == 2
        assert result[0]["type"] == "user"
        assert result[1]["type"] == "assistant"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        """read_entries skips blank and whitespace-only lines."""
        transcript = tmp_path / "transcript.jsonl"
        with open(transcript, "w") as f:
            f.write('{"type": "user"}\n')
            f.write("\n")
            f.write("   \n")
            f.write('{"type": "assistant"}\n')

        result = list(read_entries(transcript))
        assert len(result) == 2

    def test_returns_without_raising_on_ioerror(self, tmp_path: Path) -> None:
        """read_entries returns gracefully if file doesn't exist."""
        transcript = tmp_path / "nonexistent.jsonl"
        result = list(read_entries(transcript))
        assert result == []

    def test_returns_without_raising_on_permission_error(self, tmp_path: Path) -> None:
        """read_entries returns gracefully on permission errors."""
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text('{"type": "user"}\n')
        transcript.chmod(0o000)
        try:
            result = list(read_entries(transcript))
            assert result == []
        finally:
            transcript.chmod(0o644)


class TestExtractFinalAssistantText:
    """Tests for extract_final_assistant_text function."""

    def test_returns_empty_for_empty_file(self, tmp_path: Path) -> None:
        """Empty transcript returns empty string."""
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("")
        assert extract_final_assistant_text(transcript) == ""

    def test_returns_empty_for_nonexistent_file(self, tmp_path: Path) -> None:
        """Nonexistent file returns empty string without raising."""
        transcript = tmp_path / "nonexistent.jsonl"
        assert extract_final_assistant_text(transcript) == ""

    def test_extracts_text_blocks_from_assistant_entries(self, tmp_path: Path) -> None:
        """Collects text blocks from assistant entries after last user prompt."""
        transcript = tmp_path / "transcript.jsonl"
        entries = [
            {
                "type": "user",
                "message": {"role": "user", "content": "hi"},
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "sure"},
                        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
                    ],
                },
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "done"}],
                },
                "isSidechain": False,
                "isMeta": False,
            },
        ]
        with open(transcript, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = extract_final_assistant_text(transcript)
        assert result == "sure\ndone"

    def test_skips_sidechain_assistant_entries(self, tmp_path: Path) -> None:
        """Sidechain assistant entries are skipped."""
        transcript = tmp_path / "transcript.jsonl"
        entries = [
            {
                "type": "user",
                "message": {"role": "user", "content": "hi"},
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "first"}],
                },
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "sidechain"}],
                },
                "isSidechain": True,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "last"}],
                },
                "isSidechain": False,
                "isMeta": False,
            },
        ]
        with open(transcript, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = extract_final_assistant_text(transcript)
        assert result == "first\nlast"
        assert "sidechain" not in result

    def test_skips_meta_assistant_entries(self, tmp_path: Path) -> None:
        """Meta assistant entries are skipped."""
        transcript = tmp_path / "transcript.jsonl"
        entries = [
            {
                "type": "user",
                "message": {"role": "user", "content": "hi"},
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "real"}],
                },
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "meta"}],
                },
                "isSidechain": False,
                "isMeta": True,
            },
        ]
        with open(transcript, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = extract_final_assistant_text(transcript)
        assert result == "real"
        assert "meta" not in result

    def test_ignores_tool_use_blocks(self, tmp_path: Path) -> None:
        """Tool_use blocks in assistant entries are skipped."""
        transcript = tmp_path / "transcript.jsonl"
        entries = [
            {
                "type": "user",
                "message": {"role": "user", "content": "run test"},
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me run that"},
                        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
                        {"type": "text", "text": "Done!"},
                    ],
                },
                "isSidechain": False,
                "isMeta": False,
            },
        ]
        with open(transcript, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = extract_final_assistant_text(transcript)
        assert result == "Let me run that\nDone!"

    def test_ignores_thinking_blocks(self, tmp_path: Path) -> None:
        """Thinking blocks are skipped (not extracted)."""
        transcript = tmp_path / "transcript.jsonl"
        entries = [
            {
                "type": "user",
                "message": {"role": "user", "content": "think about this"},
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "text": "internal thinking"},
                        {"type": "text", "text": "response"},
                    ],
                },
                "isSidechain": False,
                "isMeta": False,
            },
        ]
        with open(transcript, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = extract_final_assistant_text(transcript)
        assert result == "response"

    def test_returns_empty_for_assistant_with_only_tool_use(self, tmp_path: Path) -> None:
        """An assistant turn with only tool_use (no text) returns empty."""
        transcript = tmp_path / "transcript.jsonl"
        entries = [
            {
                "type": "user",
                "message": {"role": "user", "content": "run"},
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}
                    ],
                },
                "isSidechain": False,
                "isMeta": False,
            },
        ]
        with open(transcript, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = extract_final_assistant_text(transcript)
        assert result == ""

    def test_uses_last_real_user_prompt(self, tmp_path: Path) -> None:
        """Only entries after the last real user prompt are returned."""
        transcript = tmp_path / "transcript.jsonl"
        entries = [
            {
                "type": "user",
                "message": {"role": "user", "content": "first prompt"},
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "first response"}],
                },
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "result",
                        }
                    ],
                },
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "user",
                "message": {"role": "user", "content": "second prompt"},
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "second response"}],
                },
                "isSidechain": False,
                "isMeta": False,
            },
        ]
        with open(transcript, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = extract_final_assistant_text(transcript)
        assert result == "second response"
        assert "first response" not in result

    def test_skips_sidechain_user_entries(self, tmp_path: Path) -> None:
        """Sidechain user entries are skipped when searching for last user prompt."""
        transcript = tmp_path / "transcript.jsonl"
        entries = [
            {
                "type": "user",
                "message": {"role": "user", "content": "real prompt"},
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "user",
                "message": {"role": "user", "content": "sidechain prompt"},
                "isSidechain": True,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "response"}],
                },
                "isSidechain": False,
                "isMeta": False,
            },
        ]
        with open(transcript, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = extract_final_assistant_text(transcript)
        assert result == "response"

    def test_skips_meta_user_entries(self, tmp_path: Path) -> None:
        """Meta user entries are skipped when searching for last user prompt."""
        transcript = tmp_path / "transcript.jsonl"
        entries = [
            {
                "type": "user",
                "message": {"role": "user", "content": "real prompt"},
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "user",
                "message": {"role": "user", "content": "meta prompt"},
                "isSidechain": False,
                "isMeta": True,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "response"}],
                },
                "isSidechain": False,
                "isMeta": False,
            },
        ]
        with open(transcript, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = extract_final_assistant_text(transcript)
        assert result == "response"

    def test_tool_result_entries_dont_count_as_user_prompt(self, tmp_path: Path) -> None:
        """User entries with array content (tool results) don't count as real prompts."""
        transcript = tmp_path / "transcript.jsonl"
        entries = [
            {
                "type": "user",
                "message": {"role": "user", "content": "first prompt"},
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "thinking..."},
                        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
                    ],
                },
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "exit 0",
                        }
                    ],
                },
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "final response"}],
                },
                "isSidechain": False,
                "isMeta": False,
            },
        ]
        with open(transcript, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = extract_final_assistant_text(transcript)
        # The tool_result entry shouldn't reset the last_user_idx, so we get response after the real prompt
        assert "final response" in result

    def test_skips_empty_text_blocks(self, tmp_path: Path) -> None:
        """Empty text blocks are skipped."""
        transcript = tmp_path / "transcript.jsonl"
        entries = [
            {
                "type": "user",
                "message": {"role": "user", "content": "hi"},
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "text", "text": "   "},
                        {"type": "text", "text": "real text"},
                    ],
                },
                "isSidechain": False,
                "isMeta": False,
            },
        ]
        with open(transcript, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = extract_final_assistant_text(transcript)
        assert result == "real text"

    def test_strips_whitespace_in_final_result(self, tmp_path: Path) -> None:
        """Final result is stripped of leading/trailing whitespace."""
        transcript = tmp_path / "transcript.jsonl"
        entries = [
            {
                "type": "user",
                "message": {"role": "user", "content": "hi"},
                "isSidechain": False,
                "isMeta": False,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "  spaced  "},
                        {"type": "text", "text": "text"},
                    ],
                },
                "isSidechain": False,
                "isMeta": False,
            },
        ]
        with open(transcript, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = extract_final_assistant_text(transcript)
        assert result == "spaced  \ntext"
        # The parts join with newline, but the final .strip() removes leading/trailing space of the joined result
        assert not result.startswith(" ")
        assert not result.endswith(" ")
