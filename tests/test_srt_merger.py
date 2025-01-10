import pytest
from unittest.mock import patch
from pathlib import Path
import datetime
from unittest.mock import Mock, create_autospec
from srt import Subtitle
from srt_labeler.merger import (
    SrtMergeError,
    SrtMerger,
    DEFAULT_VALID_LABELS,
    parse_arguments,
    main,
)


# Test SrtMergeError
def test_srtmergeerror_init_with_message():
    """Test SrtMergeError initialization with message."""
    message = "test error"
    error = SrtMergeError(message)
    assert str(error) == f"SRT merge error: {message}"


# Test SrtMerger.__init__
def test_srtmerger_init_default_values():
    """Test SrtMerger initialization with default values."""
    merger = SrtMerger()
    assert merger.valid_labels == [label.lower() for label in DEFAULT_VALID_LABELS]
    assert merger.log is not None


def test_srtmerger_init_with_custom_labels():
    """Test SrtMerger initialization with custom valid labels."""
    custom_labels = ["host", "guest"]
    merger = SrtMerger(valid_labels=custom_labels)
    assert merger.valid_labels == [label.lower() for label in custom_labels]


def test_srtmerger_init_enables_debug_logging(capsys):
    """Test SrtMerger initialization with debug mode."""
    merger = SrtMerger(debug=True)
    merger.log.debug("test debug message")
    captured = capsys.readouterr()
    assert "test debug message" in captured.err


def test_srtmerger_init_with_custom_srt_module():
    """Test SrtMerger initialization with custom srt module."""
    mock_srt = Mock()
    merger = SrtMerger(srt_module=mock_srt)
    assert merger.srt_module == mock_srt


@pytest.fixture
def mock_subtitle():
    """Fixture providing a mock subtitle with configurable content."""

    def _make_sub(content: str = "test"):
        sub = create_autospec(Subtitle, instance=True)
        sub.content = content
        sub.start = datetime.timedelta(seconds=1)
        sub.end = datetime.timedelta(seconds=2)
        sub.index = 1
        return sub

    return _make_sub


# Test _validate_subtitle_counts
def test_validatesubtitlecounts_with_matching_counts(mock_subtitle):
    """Test validation with matching subtitle counts."""
    merger = SrtMerger()
    subs1 = [mock_subtitle(), mock_subtitle()]
    subs2 = [mock_subtitle(), mock_subtitle()]
    merger._validate_subtitle_counts(subs1, subs2)  # Should not raise


def test_validatesubtitlecounts_with_empty_unlabeled(mock_subtitle):
    """Test validation with empty unlabeled file."""
    merger = SrtMerger()
    with pytest.raises(SrtMergeError, match="Input SRT content cannot be empty"):
        merger._validate_subtitle_counts([], [mock_subtitle()])


def test_validatesubtitlecounts_with_empty_labeled(mock_subtitle):
    """Test validation with empty labeled file."""
    merger = SrtMerger()
    with pytest.raises(SrtMergeError, match="Input SRT content cannot be empty"):
        merger._validate_subtitle_counts([mock_subtitle()], [])


def test_validatesubtitlecounts_with_mismatched_counts(mock_subtitle):
    """Test validation with mismatched subtitle counts."""
    merger = SrtMerger()
    with pytest.raises(SrtMergeError, match="different number of subtitles"):
        merger._validate_subtitle_counts(
            [mock_subtitle()], [mock_subtitle(), mock_subtitle()]
        )


# Test _validate_labels
@pytest.mark.parametrize(
    "content,should_raise,expected_error",
    [
        ("operator: test", False, None),
        ("test without label", False, None),
        (":test with no label", False, None),
        ("invalid: test", True, "Invalid label 'invalid'"),
        ("OPERATOR: test", False, None),
    ],
)
def test_validate_labels(mock_subtitle, content, should_raise, expected_error):
    """Test validation of various label formats."""
    merger = SrtMerger()
    sub = mock_subtitle(content)

    if should_raise:
        with pytest.raises(SrtMergeError, match=expected_error):
            merger._validate_labels([sub])
    else:
        merger._validate_labels([sub])  # Should not raise


# Test _extract_speaker
def test_extract_speaker_simple():
    """Test extraction of simple speaker label."""
    merger = SrtMerger()
    assert merger._extract_speaker("operator: test") == "operator"


def test_extract_speaker_numeric():
    """Test extraction of numeric speaker label."""
    merger = SrtMerger()
    assert merger._extract_speaker("speaker1: test") == "speaker1"
    assert merger._extract_speaker("123: test") == "123"
    assert merger._extract_speaker("speaker2: test") == "speaker2"


def test_timing_preservation():
    """Test timing information is preserved throughout the pipeline."""
    merger = SrtMerger()

    unlabeled_srt_content = """1
00:00:01,000 --> 00:00:02,000
Hello world

2
00:00:02,500 --> 00:00:03,500
How are you?
"""

    labeled_srt_content = """1
00:00:01,000 --> 00:00:02,000
operator: Hello world

2
00:00:02,500 --> 00:00:03,500
caller: How are you?
"""

    # Parse the SRT content
    subs_without = merger._parse_srt_content(unlabeled_srt_content)
    subs_with = merger._parse_srt_content(labeled_srt_content)

    # Apply labels
    labeled_subs = merger._apply_labels(subs_without, subs_with)

    # Verify timing is preserved
    for orig_sub, labeled_sub in zip(subs_without, labeled_subs):
        assert orig_sub.start == labeled_sub.start
        assert orig_sub.end == labeled_sub.end

    # Merge and verify timing in the output
    merged_content = merger.merge(unlabeled_srt_content, labeled_srt_content)
    assert "00:00:01,000 --> 00:00:02,000" in merged_content
    assert "00:00:02,500 --> 00:00:03,500" in merged_content


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("operator: test", "operator"),
        ("speaker1: test", "speaker1"),
        ("123: test", "123"),
        ("speaker2: test", "speaker2"),
        ("test without label", "Unknown"),
        ("test:without space", "test"),
        ("test: with space", "test"),
        (":test", "Unknown"),
        ("test", "Unknown"),
        ("", "Unknown"),
    ],
)
def test_extract_speaker_various_inputs(input_text, expected):
    """Test extraction of speaker labels from various input formats."""
    merger = SrtMerger()
    assert merger._extract_speaker(input_text) == expected


# Test _apply_labels
def test_applylabels_transfers_labels_correctly(mock_subtitle):
    """Test labels are applied correctly."""
    merger = SrtMerger()
    subs_without = [mock_subtitle("test content")]
    subs_with = [mock_subtitle("operator: test content")]
    result = merger._apply_labels(subs_without, subs_with)
    assert result[0].content == "operator: test content"


def test_applylabels_preserves_original_content(mock_subtitle):
    """Test original content is preserved."""
    merger = SrtMerger()
    original_content = "test content"
    subs_without = [mock_subtitle(original_content)]
    subs_with = [mock_subtitle("operator: different content")]
    result = merger._apply_labels(subs_without, subs_with)
    assert result[0].content.endswith(original_content)


def test_applylabels_handles_empty_content(mock_subtitle):
    """Test handling of empty content."""
    merger = SrtMerger()
    subs_without = [mock_subtitle("")]
    subs_with = [mock_subtitle("operator: ")]
    result = merger._apply_labels(subs_without, subs_with)
    assert result[0].content == "operator: "


def test_apply_labels_preserves_timing(mock_subtitle):
    """Test subtitle timing is preserved."""
    merger = SrtMerger()
    start = datetime.timedelta(seconds=1)
    end = datetime.timedelta(seconds=2)
    sub = mock_subtitle("test content")
    sub.start = start
    sub.end = end
    subs_without = [sub]
    subs_with = [mock_subtitle("operator: test content")]
    result = merger._apply_labels(subs_without, subs_with)
    assert result[0].start == start
    assert result[0].end == end


# Test _parse_srt_content
def test_parsesrtcontent_with_valid_input():
    """Test parsing valid SRT content."""
    merger = SrtMerger()
    content = """1
00:00:01,000 --> 00:00:02,000
test

"""
    result = merger._parse_srt_content(content)
    assert len(result) == 1
    assert result[0].content == "test"
    assert result[0].start == datetime.timedelta(seconds=1)
    assert result[0].end == datetime.timedelta(seconds=2)


def test_parsesrtcontent_with_malformed_input():
    """Test parsing malformed SRT content."""
    merger = SrtMerger()
    mock_srt = Mock()
    mock_srt.SRTParseError = Exception
    mock_srt.parse.side_effect = Exception("Invalid format")
    merger.srt_module = mock_srt

    with pytest.raises(SrtMergeError, match="Invalid SRT format"):
        merger._parse_srt_content("invalid content")


def test_parsesrtcontent_with_empty_input():
    """Test parsing empty SRT content."""
    merger = SrtMerger()
    mock_srt = Mock()
    mock_srt.parse.return_value = []
    merger.srt_module = mock_srt

    result = merger._parse_srt_content("")
    assert len(result) == 0


def test_parse_srt_content_preserves_timing():
    """Test parsing preserves timing information."""
    merger = SrtMerger()
    content = """1
00:00:03,000 --> 00:00:04,000
test

"""
    result = merger._parse_srt_content(content)
    assert result[0].start == datetime.timedelta(seconds=3)
    assert result[0].end == datetime.timedelta(seconds=4)


# Test read_file
@pytest.mark.parametrize(
    "content,encoding,expected,should_raise",
    [
        ("test content", "utf-8", "test content", False),
        (
            "test\ncontent\nwith\nnewlines",
            "utf-8",
            "test\ncontent\nwith\nnewlines",
            False,
        ),
        (b"\xff\xfe\xfd", None, None, True),  # Invalid encoding
        (None, "utf-8", None, True),  # Missing file
    ],
)
def test_read_file(tmp_path, content, encoding, expected, should_raise):
    """Test reading files with various content and encodings."""
    merger = SrtMerger()
    test_file = tmp_path / "test.srt"

    if isinstance(content, str):
        test_file.write_text(content, encoding=encoding)
    elif content is not None:
        test_file.write_bytes(content)
    else:
        test_file = Path("/nonexistent/file.srt")

    if should_raise:
        with pytest.raises(SrtMergeError, match="Failed to read file"):
            merger.read_file(test_file)
    else:
        result = merger.read_file(test_file)
        assert result == expected


# Test merge
def test_merge_with_valid_inputs():
    """Test successful merge of SRT files."""
    merger = SrtMerger()

    unlabeled_srt = """1
00:00:01,000 --> 00:00:02,000
Hello world

2
00:00:02,500 --> 00:00:03,500
How are you?

"""

    labeled_srt = """1
00:00:01,000 --> 00:00:02,000
operator: Hi there world

2
00:00:02,500 --> 00:00:03,500
caller: How are you?

"""

    result = merger.merge(unlabeled_srt, labeled_srt)
    expected_output = """1
00:00:01,000 --> 00:00:02,000
operator: Hello world

2
00:00:02,500 --> 00:00:03,500
caller: How are you?

"""
    assert result == expected_output


def test_merge_with_invalid_label_format(mock_subtitle):
    """Test merge with invalid label format."""
    merger = SrtMerger()
    mock_srt = Mock()
    mock_srt.parse.return_value = [mock_subtitle("invalid: test")]
    merger.srt_module = mock_srt

    with pytest.raises(SrtMergeError, match="Invalid label 'invalid'"):
        merger.merge("unlabeled", "labeled")


def test_merge_with_count_mismatch(mock_subtitle):
    """Test merge with count mismatch."""
    merger = SrtMerger()
    mock_srt = Mock()
    mock_srt.parse.side_effect = [
        [mock_subtitle()],  # unlabeled
        [mock_subtitle(), mock_subtitle()],  # labeled
    ]
    merger.srt_module = mock_srt

    with pytest.raises(SrtMergeError, match="different number of subtitles"):
        merger.merge("unlabeled", "labeled")


def test_merge_preserves_timing():
    """Test merge preserves timing information."""
    merger = SrtMerger()

    unlabeled_srt = """1
00:00:05,000 --> 00:00:10,000
test content

"""

    labeled_srt = """1
00:00:05,000 --> 00:00:10,000
operator: test content

"""

    result = merger.merge(unlabeled_srt, labeled_srt)
    expected_output = """1
00:00:05,000 --> 00:00:10,000
operator: test content

"""
    assert result == expected_output


def test_merge_with_empty_input():
    """Test merge with empty input."""
    merger = SrtMerger()
    mock_srt = Mock()
    mock_srt.parse.return_value = []
    merger.srt_module = mock_srt

    with pytest.raises(SrtMergeError, match="Empty SRT file provided"):
        merger.merge("", "")


# Test parse_arguments
def test_parsearguments_with_required_args():
    """Test parsing required arguments."""
    test_args = ["unlabeled.srt", "labeled.srt"]
    with patch("sys.argv", ["script.py"] + test_args):
        args = parse_arguments()
        assert args.unlabeled_srt == Path("unlabeled.srt")
        assert args.labeled_srt == Path("labeled.srt")
        assert args.valid_labels == DEFAULT_VALID_LABELS
        assert args.debug is False


def test_parsearguments_with_custom_labels():
    """Test parsing custom valid labels."""
    test_args = [
        "unlabeled.srt",
        "labeled.srt",
        "--valid-labels",
        "host,guest,moderator",
    ]
    with patch("sys.argv", ["script.py"] + test_args):
        args = parse_arguments()
        assert args.valid_labels == ["host", "guest", "moderator"]


def test_parsearguments_with_custom_labels_with_spaces():
    """Test parsing custom valid labels with spaces."""
    test_args = [
        "unlabeled.srt",
        "labeled.srt",
        "--valid-labels",
        " host , guest , moderator ",
    ]
    with patch("sys.argv", ["script.py"] + test_args):
        args = parse_arguments()
        assert args.valid_labels == ["host", "guest", "moderator"]


def test_parsearguments_enables_debug_mode():
    """Test parsing debug mode flag."""
    test_args = [
        "unlabeled.srt",
        "labeled.srt",
        "--debug",
    ]
    with patch("sys.argv", ["script.py"] + test_args):
        args = parse_arguments()
        assert args.debug is True


def test_parsearguments_with_missing_required_args():
    """Test missing required arguments."""
    with pytest.raises(SystemExit):
        parse_arguments([])


# Test main
def test_main_with_valid_inputs(capsys, tmp_path):
    """Test successful main execution."""
    unlabeled = tmp_path / "unlabeled.srt"
    labeled = tmp_path / "labeled.srt"
    unlabeled.write_text("1\n00:00:01,000 --> 00:00:02,000\ntest\n\n")
    labeled.write_text("1\n00:00:01,000 --> 00:00:02,000\noperator: test\n\n")

    test_args = [str(unlabeled), str(labeled)]
    with patch("sys.argv", ["script.py"] + test_args):
        main()

    captured = capsys.readouterr()
    assert "operator: test" in captured.out


def test_main_outputs_merged_content_to_stdout(capsys, tmp_path):
    """Test main outputs merged content to stdout."""
    unlabeled = tmp_path / "unlabeled.srt"
    labeled = tmp_path / "labeled.srt"
    unlabeled.write_text("1\n00:00:01,000 --> 00:00:02,000\ntest\n\n")
    labeled.write_text("1\n00:00:01,000 --> 00:00:02,000\noperator: test\n\n")

    test_args = [str(unlabeled), str(labeled)]
    with patch("sys.argv", ["script.py"] + test_args):
        main()

    captured = capsys.readouterr()
    assert captured.out.strip()  # Check stdout has content
    assert not captured.err  # Check stderr is empty


def test_integration_full_workflow(tmp_path):
    """Test full workflow from file reading to merged output."""
    unlabeled = tmp_path / "unlabeled.srt"
    labeled = tmp_path / "labeled.srt"

    # Create test files
    unlabeled_content = """1
00:00:01,000 --> 00:00:02,000
Hello world

2
00:00:03,000 --> 00:00:04,000
How are you?
"""
    labeled_content = """1
00:00:01,000 --> 00:00:02,000
operator: Hi there world

2
00:00:03,000 --> 00:00:04,000
caller: How are you?
"""
    unlabeled.write_text(unlabeled_content)
    labeled.write_text(labeled_content)

    # Run merge
    merger = SrtMerger()
    result = merger.merge(unlabeled.read_text(), labeled.read_text())

    # Verify exact output
    expected_output = """1
00:00:01,000 --> 00:00:02,000
operator: Hello world

2
00:00:03,000 --> 00:00:04,000
caller: How are you?

"""
    assert result == expected_output
