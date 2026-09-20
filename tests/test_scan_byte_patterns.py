"""Tests for the byte-pattern scan module and /api/v1/scan-byte-patterns."""

import base64

import pytest
from fastapi.testclient import TestClient

from app.hex_parser import Segment
from app.main import app
from app.pattern_scan import (
    MAX_PATTERN_BYTES,
    PatternSpec,
    compile_pattern,
    compile_patterns,
    scan_patterns,
)

client = TestClient(app)


def rec(rectype: int, addr: int, data: bytes = b"") -> str:
    """Build a valid Intel HEX record line."""
    body = bytes([len(data), (addr >> 8) & 0xFF, addr & 0xFF, rectype]) + bytes(data)
    checksum = (-sum(body)) & 0xFF
    return ":" + (body + bytes([checksum])).hex().upper()


EOF = rec(0x01, 0)


def data_rec(addr: int, data: bytes) -> str:
    return rec(0x00, addr, data)


def linear_base(value: int) -> str:
    return rec(0x04, 0, value.to_bytes(2, "big"))


def spec(source: str, **kw) -> PatternSpec:
    kw.setdefault("name", "p")
    kw.setdefault("index", 0)
    return PatternSpec(source=source, **kw)


# ---------------------------------------------------------------------------
# pattern compilation (pure logic)
# ---------------------------------------------------------------------------

def test_exact_bytes_and_wildcards_compile_to_value_mask():
    pattern, errors = compile_pattern(spec("DE ad ??"))
    assert errors == []
    assert pattern.values == b"\xDE\xAD\x00"
    assert pattern.masks == b"\xFF\xFF\x00"
    assert pattern.length == 3


def test_repeat_count_expands_bytes_and_wildcards():
    pattern, errors = compile_pattern(spec("AB{3} ??{2} 0C"))
    assert errors == []
    assert pattern.values == b"\xAB\xAB\xAB\x00\x00\x0C"
    assert pattern.masks == b"\xFF\xFF\xFF\x00\x00\xFF"


def test_repeat_of_one_is_allowed():
    pattern, errors = compile_pattern(spec("AB{1}"))
    assert errors == []
    assert pattern.values == b"\xAB"


@pytest.mark.parametrize(
    "source",
    [
        "",            # empty
        "   \t  ",     # only whitespace
        "GG",          # not hex
        "0x1F",        # prefix not allowed
        "DEA",         # tokens are exactly two digits
        "D",           # single hex digit
        "?",           # single question mark
        "???",         # wildcard is exactly two question marks
        "DE{0}",       # repeat must be positive
        "DE{}",        # empty repeat
        "DE{x}",       # non-numeric repeat
        "DE{2",        # unbalanced brace
        "{2}DE",       # repeat must follow a byte
        "DE{2}{3}",    # one repeat per token
        "DEAD",        # tokens must be whitespace-separated
    ],
)
def test_invalid_pattern_syntax_is_rejected(source):
    pattern, errors = compile_pattern(spec(source))
    assert pattern is None
    assert [e.kind for e in errors] == ["invalid_pattern_syntax"]
    assert errors[0].name == "p" and errors[0].index == 0


def test_several_bad_tokens_are_all_reported():
    pattern, errors = compile_pattern(spec("GG ?? HH{0}"))
    assert pattern is None
    assert [e.kind for e in errors] == ["invalid_pattern_syntax"] * 2


def test_compiled_length_is_capped():
    pattern, errors = compile_pattern(spec(f"AA{{{MAX_PATTERN_BYTES}}}"))
    assert errors == [] and pattern.length == MAX_PATTERN_BYTES

    pattern, errors = compile_pattern(spec(f"??{{{MAX_PATTERN_BYTES + 1}}}"))
    assert pattern is None
    assert [e.kind for e in errors] == ["pattern_too_long"]

    pattern, errors = compile_pattern(spec(f"AA{{2000}} BB{{{MAX_PATTERN_BYTES - 1999}}}"))
    assert pattern is None
    assert [e.kind for e in errors] == ["pattern_too_long"]


def test_range_must_be_given_together_and_be_valid():
    _, errors = compile_pattern(spec("AA", range_start=8))
    assert [e.kind for e in errors] == ["invalid_range"]
    _, errors = compile_pattern(spec("AA", range_end=8))
    assert [e.kind for e in errors] == ["invalid_range"]
    _, errors = compile_pattern(spec("AA", range_start=8, range_end=8))
    assert [e.kind for e in errors] == ["invalid_range"]
    _, errors = compile_pattern(spec("AA", range_start=16, range_end=8))
    assert [e.kind for e in errors] == ["invalid_range"]
    _, errors = compile_pattern(spec("AA", range_start=-4, range_end=8))
    assert [e.kind for e in errors] == ["invalid_range"]
    _, errors = compile_pattern(spec("AA", range_start=0, range_end=(1 << 32) + 1))
    assert [e.kind for e in errors] == ["invalid_range"]
    # the very end of the 32-bit space is allowed
    pattern, errors = compile_pattern(spec("AA", range_start=0, range_end=1 << 32))
    assert errors == [] and pattern.range_end == 1 << 32


def test_alignment_must_be_positive():
    _, errors = compile_pattern(spec("AA", alignment=0))
    assert [e.kind for e in errors] == ["invalid_alignment"]
    _, errors = compile_pattern(spec("AA", alignment=-4))
    assert [e.kind for e in errors] == ["invalid_alignment"]


def test_expected_count_must_not_be_negative():
    _, errors = compile_pattern(spec("AA", expected_count=-1))
    assert [e.kind for e in errors] == ["invalid_expected_count"]
    pattern, errors = compile_pattern(spec("AA", expected_count=0))
    assert errors == [] and pattern.expected_count == 0


def test_all_problem_classes_are_aggregated_per_pattern():
    _, errors = compile_pattern(
        spec("GG", range_start=9, range_end=4, alignment=0, expected_count=-2)
    )
    assert [e.kind for e in errors] == [
        "invalid_pattern_syntax",
        "invalid_range",
        "invalid_alignment",
        "invalid_expected_count",
    ]


def test_compile_patterns_aggregates_across_specs_in_order():
    specs = [
        PatternSpec(name="first", index=0, source="GG"),
        PatternSpec(name="ok", index=1, source="AA"),
        PatternSpec(name="second", index=2, source="BB", alignment=0),
    ]
    compiled, errors = compile_patterns(specs)
    assert [c.name for c in compiled] == ["ok"]
    assert [(e.name, e.kind) for e in errors] == [
        ("first", "invalid_pattern_syntax"),
        ("second", "invalid_alignment"),
    ]
    assert [e.index for e in errors] == [0, 2]


# ---------------------------------------------------------------------------
# scanning (pure logic)
# ---------------------------------------------------------------------------

def scan(source: str, segments: list[Segment], **kw):
    pattern, errors = compile_pattern(spec(source, **kw))
    assert errors == []
    return scan_patterns(segments, [pattern])


def test_basic_and_overlapping_hits():
    result = scan("02", [Segment(0, b"\x01\x02\x03\x02\x01")])
    assert result.reports[0].status == "ambiguous"
    assert result.reports[0].match_addresses == [1, 3]

    result = scan("AA AA", [Segment(0, b"\xAA\xAA\xAA")])
    assert [m.address for m in result.matches] == [0, 1]  # overlapping
    assert result.reports[0].status == "ambiguous"


def test_wildcard_matches_any_byte():
    result = scan("01 ?? 03", [Segment(0, b"\x01\x02\x03")])
    assert result.reports[0].status == "unique"
    assert result.matches[0].address == 0
    assert result.matches[0].data == b"\x01\x02\x03"


def test_matches_never_cross_segment_gaps():
    segments = [Segment(0, b"\x01\x02"), Segment(4, b"\x03\x04")]
    result = scan("02 03", segments)
    assert result.reports[0].status == "absent"
    assert result.matches == []
    # the bytes themselves are still found inside their own segments
    assert scan("02", segments).reports[0].match_addresses == [1]
    assert scan("03", segments).reports[0].match_addresses == [4]


def test_range_restricts_hit_start_addresses():
    segments = [Segment(0, b"\x01\x02\x03\x02")]
    result = scan("02", segments, range_start=2, range_end=4)
    assert result.reports[0].match_addresses == [3]
    # a match may extend past range_end; only the start address is constrained
    result = scan("02 03", segments, range_start=0, range_end=2)
    assert result.reports[0].match_addresses == [1]
    # range outside every segment yields no hits
    result = scan("02", segments, range_start=100, range_end=200)
    assert result.reports[0].status == "absent"


def test_alignment_restricts_hit_start_addresses():
    segments = [Segment(0, b"\x02" * 8)]
    result = scan("02", segments, alignment=4)
    assert result.reports[0].match_addresses == [0, 4]
    # alignment applies to absolute addresses, not segment offsets
    result = scan("02", [Segment(2, b"\x02" * 4)], alignment=4)
    assert result.reports[0].match_addresses == [4]
    # overlapping hits survive a small alignment
    result = scan("02 02 02", [Segment(0, b"\x02" * 5)], alignment=2)
    assert result.reports[0].match_addresses == [0, 2]


def test_statuses_follow_hit_count_and_expected_count():
    segments = [Segment(0, b"\x01\x02\x03\x02\x01")]
    assert scan("09", segments).reports[0].status == "absent"
    assert scan("03", segments).reports[0].status == "unique"
    assert scan("02", segments).reports[0].status == "ambiguous"
    # expected count met: status falls back to the hit-count class
    assert scan("02", segments, expected_count=2).reports[0].status == "ambiguous"
    assert scan("03", segments, expected_count=1).reports[0].status == "unique"
    assert scan("09", segments, expected_count=0).reports[0].status == "absent"
    # expected count missed
    report = scan("02", segments, expected_count=1).reports[0]
    assert report.status == "count_mismatch" and report.hit_count == 2
    report = scan("09", segments, expected_count=1).reports[0]
    assert report.status == "count_mismatch" and report.hit_count == 0


def test_ok_is_false_only_on_count_mismatch():
    segments = [Segment(0, b"\x01\x02")]
    assert scan("09", segments).ok is True  # absent without expectation
    assert scan("09", segments, expected_count=1).ok is False


def test_matches_are_sorted_by_address_then_pattern_order():
    specs = [
        PatternSpec(name="a", index=0, source="05"),
        PatternSpec(name="b", index=1, source="03"),
        PatternSpec(name="c", index=2, source="05"),
    ]
    compiled, errors = compile_patterns(specs)
    assert errors == []
    result = scan_patterns([Segment(0, b"\x05\x03\x05")], compiled)
    # sorted by (address, pattern index)
    assert [(m.address, m.pattern_index) for m in result.matches] == [
        (0, 0),
        (0, 2),
        (1, 1),
        (2, 0),
        (2, 2),
    ]


def test_pattern_longer_than_segment_finds_nothing():
    result = scan("01 02 03 04 05", [Segment(0, b"\x01\x02")])
    assert result.reports[0].status == "absent"


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

def test_api_happy_path_statuses_and_ordering():
    text = "\n".join([
        linear_base(0x0800),
        data_rec(0x0000, bytes.fromhex("214601360121470136007EFE09D21901")),
        rec(0x05, 0, (0x080001C1).to_bytes(4, "big")),
        EOF,
    ])
    resp = client.post(
        "/api/v1/scan-byte-patterns",
        json={
            "hex_text": text,
            "patterns": [
                {"name": "thumb_branch", "pattern": "01 36", "expected_count": 2},
                {
                    "name": "wildcard_pair",
                    "pattern": "7E ?? 09",
                    "range_start": 0x08000000,
                    "range_end": 0x08000010,
                },
                {"name": "padding", "pattern": "FF{4}"},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["hex_ok"] is True
    assert body["patterns_valid"] is True
    assert body["error_count"] == 0
    assert body["entry_address"] == 0x080001C1
    assert body["total_bytes"] == 16
    assert body["pattern_count"] == 3
    assert body["match_count"] == 3
    assert body["pattern_errors"] == []

    matches = body["matches"]
    assert [(m["pattern"], m["address"]) for m in matches] == [
        ("thumb_branch", 0x08000002),
        ("thumb_branch", 0x08000007),
        ("wildcard_pair", 0x0800000A),
    ]
    assert matches[0]["end"] == 0x08000004 and matches[0]["size"] == 2
    assert base64.b64decode(matches[0]["data_base64"]) == b"\x01\x36"
    assert base64.b64decode(matches[2]["data_base64"]) == b"\x7E\xFE\x09"
    assert [m["pattern_index"] for m in matches] == [0, 0, 1]

    patterns = body["patterns"]
    assert [p["name"] for p in patterns] == ["thumb_branch", "wildcard_pair", "padding"]
    assert [p["status"] for p in patterns] == ["ambiguous", "unique", "absent"]
    assert patterns[0]["hit_count"] == 2 and patterns[0]["expected_count"] == 2
    assert patterns[0]["match_addresses"] == [0x08000002, 0x08000007]
    assert patterns[1]["compiled_length"] == 3
    assert patterns[1]["range_start"] == 0x08000000
    assert patterns[1]["range_end"] == 0x08000010
    assert patterns[1]["alignment"] == 1
    assert patterns[2]["hit_count"] == 0 and patterns[2]["match_addresses"] == []
    # parse diagnostics of the shared parser are preserved
    assert any(d["record_type"] == "extended_linear_address" for d in body["diagnostics"])


def test_api_match_never_crosses_a_gap():
    text = "\n".join([
        data_rec(0x0000, b"\x01\x02"),
        data_rec(0x0004, b"\x03\x04"),
        EOF,
    ])
    body = client.post(
        "/api/v1/scan-byte-patterns",
        json={"hex_text": text, "patterns": [{"name": "span", "pattern": "02 03"}]},
    ).json()
    assert body["ok"] is True
    assert body["patterns"][0]["status"] == "absent"
    assert body["matches"] == []


def test_api_count_mismatch_fails_ok_but_reports_hits():
    text = "\n".join([data_rec(0x0000, b"\x02\x02\x02"), EOF])
    body = client.post(
        "/api/v1/scan-byte-patterns",
        json={
            "hex_text": text,
            "patterns": [{"name": "pair", "pattern": "02 02", "expected_count": 1}],
        },
    ).json()
    assert body["ok"] is False
    assert body["hex_ok"] is True and body["patterns_valid"] is True
    report = body["patterns"][0]
    assert report["status"] == "count_mismatch"
    assert report["hit_count"] == 2  # overlapping hits
    assert [m["address"] for m in body["matches"]] == [0, 1]


def test_api_invalid_hex_and_invalid_patterns_are_aggregated_without_scan():
    resp = client.post(
        "/api/v1/scan-byte-patterns",
        json={
            "hex_text": ":00000001FE\n",  # bad checksum, no EOF
            "patterns": [
                {"name": "bad_syntax", "pattern": "GG"},
                {"name": "ok", "pattern": "AA"},
                {"name": "bad_range", "pattern": "BB", "range_start": 9, "range_end": 4},
                {"name": "bad_align", "pattern": "CC", "alignment": 0},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["hex_ok"] is False
    assert body["patterns_valid"] is False
    assert body["matches"] == [] and body["patterns"] == []
    assert body["match_count"] == 0
    assert body["error_count"] >= 1  # HEX diagnostics are still returned
    kinds = [(e["pattern"], e["kind"]) for e in body["pattern_errors"]]
    assert kinds == [
        ("bad_syntax", "invalid_pattern_syntax"),
        ("bad_range", "invalid_range"),
        ("bad_align", "invalid_alignment"),
    ]
    assert [e["index"] for e in body["pattern_errors"]] == [0, 2, 3]


def test_api_invalid_hex_with_valid_patterns_marks_patterns_valid():
    body = client.post(
        "/api/v1/scan-byte-patterns",
        json={
            "hex_text": ":00000001FE\n",
            "patterns": [{"name": "ok", "pattern": "AA ??"}],
        },
    ).json()
    assert body["ok"] is False
    assert body["hex_ok"] is False
    assert body["patterns_valid"] is True
    assert body["pattern_errors"] == []
    assert body["matches"] == [] and body["patterns"] == []


def test_api_invalid_pattern_with_valid_hex_marks_hex_ok():
    text = "\n".join([data_rec(0x0000, b"\x01"), EOF])
    body = client.post(
        "/api/v1/scan-byte-patterns",
        json={
            "hex_text": text,
            "patterns": [{"name": "bad", "pattern": "AA{0}"}],
        },
    ).json()
    assert body["ok"] is False
    assert body["hex_ok"] is True
    assert body["patterns_valid"] is False
    assert body["pattern_errors"][0]["kind"] == "invalid_pattern_syntax"
    assert body["matches"] == [] and body["patterns"] == []


def test_api_expected_count_zero_is_honored():
    text = "\n".join([data_rec(0x0000, b"\x01\x02"), EOF])
    body = client.post(
        "/api/v1/scan-byte-patterns",
        json={
            "hex_text": text,
            "patterns": [
                {"name": "gone", "pattern": "FF", "expected_count": 0},
                {"name": "here", "pattern": "01", "expected_count": 0},
            ],
        },
    ).json()
    assert body["ok"] is False
    assert body["patterns"][0]["status"] == "absent"  # expectation met
    assert body["patterns"][1]["status"] == "count_mismatch"


@pytest.mark.parametrize(
    "payload",
    [
        {"patterns": []},  # at least one pattern
        {"patterns": [{"name": "a", "pattern": "AA"}, {"name": "a", "pattern": "BB"}]},
        {"patterns": [{"name": "", "pattern": "AA"}]},
        {"patterns": [{"name": "a"}]},  # missing pattern text
        {"patterns": [{"pattern": "AA"}]},  # missing name
        {"hex_text": 123, "patterns": [{"name": "a", "pattern": "AA"}]},
    ],
)
def test_api_schema_rejects_bad_requests(payload):
    request = {
        "hex_text": "\n".join([EOF]),
        "patterns": [{"name": "a", "pattern": "AA"}],
    }
    request.update(payload)
    resp = client.post("/api/v1/scan-byte-patterns", json=request)
    assert resp.status_code == 422


def test_api_alignment_and_range_filters_via_http():
    text = "\n".join([data_rec(0x0000, b"\x02" * 16), EOF])
    body = client.post(
        "/api/v1/scan-byte-patterns",
        json={
            "hex_text": text,
            "patterns": [
                {
                    "name": "aligned",
                    "pattern": "02 02",
                    "range_start": 4,
                    "range_end": 12,
                    "alignment": 4,
                }
            ],
        },
    ).json()
    assert body["ok"] is True
    assert body["patterns"][0]["match_addresses"] == [4, 8]
    assert body["patterns"][0]["status"] == "ambiguous"
