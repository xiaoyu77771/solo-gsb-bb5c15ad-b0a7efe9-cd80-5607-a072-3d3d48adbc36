"""Tests for the per-page CRC-32 injection module and /api/v1/inject-page-crc."""

import base64
import zlib

import pytest
from fastapi.testclient import TestClient

from app.hex_parser import Segment, parse_intel_hex
from app.main import app
from app.page_crc import (
    MAX_PAGES,
    TABLE_ENTRY_SIZE,
    inject_page_crc,
    validate_page_crc_params,
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


def linear_entry(value: int) -> str:
    return rec(0x05, 0, value.to_bytes(4, "big"))


def materialize(memory: dict[int, int], start: int, end: int, table: tuple[int, int]) -> bytes:
    """Independently materialize a page: image bytes, 0xFF holes, table area 0xFF."""
    table_start, table_end = table
    return bytes(
        0xFF if table_start <= addr < table_end else memory.get(addr, 0xFF)
        for addr in range(start, end)
    )


def crc32hex(data: bytes) -> str:
    """The IEEE CRC-32 of ``data`` as 8 lowercase hex digits."""
    return f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"


# ---------------------------------------------------------------------------
# parameter validation (pure logic)
# ---------------------------------------------------------------------------

def test_valid_params_have_no_errors():
    assert validate_page_crc_params([], 0, 16, 8, 8) == []


def test_range_must_be_a_non_empty_half_open_interval():
    errors = validate_page_crc_params([], 16, 16, 8, 0)
    assert [e.kind for e in errors] == ["invalid_range"]
    errors = validate_page_crc_params([], 32, 16, 8, 0)
    assert [e.kind for e in errors] == ["invalid_range"]


def test_range_end_may_not_leave_32_bit_space():
    errors = validate_page_crc_params([], 0, (1 << 32) + 16, 16, 0)
    assert [e.kind for e in errors] == ["invalid_range"]
    # the very end of the 32-bit space is allowed when aligned
    assert validate_page_crc_params([], 0xFFFFFFF0, 1 << 32, 16, 0xFFFFFFF0) == []


def test_both_endpoints_must_be_page_aligned():
    errors = validate_page_crc_params([], 1, 16, 8, 0)
    assert [e.kind for e in errors] == ["range_unaligned"]
    errors = validate_page_crc_params([], 0, 15, 8, 0)
    assert [e.kind for e in errors] == ["range_unaligned"]
    errors = validate_page_crc_params([], 2, 14, 8, 0)
    assert [e.kind for e in errors] == ["range_unaligned"]


def test_page_count_is_capped_at_4096():
    page_size = 4
    assert validate_page_crc_params([], 0, MAX_PAGES * page_size, page_size, 0) == []
    errors = validate_page_crc_params([], 0, (MAX_PAGES + 1) * page_size, page_size, 0)
    assert [e.kind for e in errors] == ["too_many_pages"]
    assert str(MAX_PAGES) in errors[0].message


def test_table_address_must_be_four_byte_aligned():
    errors = validate_page_crc_params([], 0, 16, 8, 2)
    assert [e.kind for e in errors] == ["table_address_unaligned"]
    assert errors[0].address == 2
    # a misaligned table address is reported even when the range is broken
    errors = validate_page_crc_params([], 8, 8, 8, 10)
    assert {e.kind for e in errors} == {"invalid_range", "table_address_unaligned"}


def test_table_must_lie_wholly_inside_the_range():
    # table end past range_end
    errors = validate_page_crc_params([], 0, 16, 8, 12)
    assert [e.kind for e in errors] == ["table_out_of_range"]
    assert errors[0].start == 12 and errors[0].end == 12 + 2 * TABLE_ENTRY_SIZE
    # table start below range_start
    errors = validate_page_crc_params([], 16, 32, 8, 8)
    assert [e.kind for e in errors] == ["table_out_of_range"]
    # negative table address: aligned, but outside the range
    errors = validate_page_crc_params([], 0, 16, 8, -8)
    assert [e.kind for e in errors] == ["table_out_of_range"]


def test_table_area_must_be_unoccupied():
    segments = [Segment(0, bytes(range(16)))]
    errors = validate_page_crc_params(segments, 0, 16, 8, 8)
    assert [e.kind for e in errors] == ["table_region_occupied"]
    assert errors[0].start == 8 and errors[0].end == 16

    # every maximal contiguous occupied run is reported separately
    memory_segments = [Segment(8, b"\x01\x02"), Segment(13, b"\x03")]
    errors = validate_page_crc_params(memory_segments, 0, 16, 8, 8)
    assert [e.kind for e in errors] == ["table_region_occupied"] * 2
    assert [(e.start, e.end) for e in errors] == [(8, 10), (13, 14)]


def test_several_error_classes_are_reported_together():
    errors = validate_page_crc_params([], 0, 16, 8, 14)  # misaligned and out of range
    assert {e.kind for e in errors} == {"table_address_unaligned", "table_out_of_range"}


def test_param_failure_injects_nothing():
    result = inject_page_crc([Segment(0, bytes(16))], 0, 16, 8, 8)
    assert not result.ok
    assert result.params_valid is False
    assert result.pages == [] and result.table == b"" and result.segments == []
    assert result.final_total_bytes == 0
    assert result.page_count == 2  # would-be page count is still echoed
    assert result.table_size == 2 * TABLE_ENTRY_SIZE
    assert [e.kind for e in result.param_errors] == ["table_region_occupied"]


# ---------------------------------------------------------------------------
# CRC computation and table injection (pure logic)
# ---------------------------------------------------------------------------

def test_crc32_is_the_ieee_checksum():
    # zlib.crc32 is the reflected 0xEDB88320 IEEE CRC-32; anchor the choice
    # against the classic check vector.
    assert zlib.crc32(b"123456789") & 0xFFFFFFFF == 0xCBF43926
    result = inject_page_crc([Segment(0, b"123456789")], 0, 16, 16, 12)
    assert result.ok
    assert result.pages[0].crc32 == crc32hex(b'123456789' + b'\xff' * 7)


def test_two_page_happy_path_materializes_and_merges():
    segments = [Segment(0, bytes(range(8)))]
    result = inject_page_crc(segments, 0, 16, 8, 8)
    assert result.ok and result.params_valid and result.param_errors == []
    assert result.page_count == 2
    assert result.table_size == 8 and result.table_end == 16

    page0, page1 = result.pages
    assert (page0.index, page0.start, page0.end, page0.size) == (0, 0, 8, 8)
    assert page0.present_bytes == 8 and page0.table_bytes == 0
    assert page0.crc32 == crc32hex(bytes(range(8)))
    assert (page1.index, page1.start, page1.end, page1.size) == (1, 8, 16, 8)
    assert page1.present_bytes == 0 and page1.table_bytes == 8
    # the page holding the table is checksummed as all 0xFF
    assert page1.crc32 == crc32hex(b'\xff' * 8)

    # table entries are little-endian, in page order
    expected = bytes.fromhex(page0.crc32)[::-1] + bytes.fromhex(page1.crc32)[::-1]
    assert result.table == expected
    assert int.from_bytes(result.table[:4], "little") == int(page0.crc32, 16)
    assert int.from_bytes(result.table[4:], "little") == int(page1.crc32, 16)

    # the table merged into the image closes the gap between the segments
    assert len(result.segments) == 1
    merged = result.segments[0]
    assert merged.start == 0 and merged.end == 16
    assert merged.data == bytes(range(8)) + expected
    assert result.final_total_bytes == 16


def test_table_area_is_forced_to_ff_inside_a_shared_page():
    segments = [Segment(0, b"\x01\x02\x03\x04")]
    result = inject_page_crc(segments, 0, 16, 16, 4)
    assert result.ok
    (page,) = result.pages
    assert page.table_bytes == 4 and page.present_bytes == 4
    expected = b"\x01\x02\x03\x04" + b"\xff" * 12  # table at [4, 8) reads as 0xFF
    assert page.crc32 == crc32hex(expected)
    # the injected table extends the explicit image to the end of the table
    (merged,) = result.segments
    assert merged.start == 0 and merged.end == 8
    assert merged.data == b"\x01\x02\x03\x04" + result.table
    assert merged.data[4:8] == result.table


def test_sparse_image_and_bytes_outside_the_range():
    segments = [Segment(0, b"\x01\x02"), Segment(12, b"\x03"), Segment(1000, b"\xEE")]
    result = inject_page_crc(segments, 0, 32, 8, 16)
    assert result.ok
    assert result.page_count == 4 and result.table_size == 16
    memory = {0: 1, 1: 2, 12: 3}
    for page in result.pages:
        expected = materialize(memory, page.start, page.end, (16, 32))
        assert page.crc32 == crc32hex(expected)
    assert [p.present_bytes for p in result.pages] == [2, 1, 0, 0]
    assert [p.table_bytes for p in result.pages] == [0, 0, 8, 8]
    # the byte outside the range does not participate but survives the merge
    assert result.segments[-1].start == 1000
    assert result.final_total_bytes == 4 + 16


def test_empty_image_yields_ff_pages_and_a_table_only_image():
    result = inject_page_crc([], 0, 16, 8, 8)
    assert result.ok
    assert [p.present_bytes for p in result.pages] == [0, 0]
    assert all(p.crc32 == crc32hex(b'\xff' * 8) for p in result.pages)
    assert len(result.segments) == 1
    assert result.segments[0].start == 8 and result.segments[0].data == result.table
    assert result.final_total_bytes == 8


def test_injected_image_verifies_against_the_returned_crcs():
    # Emulate the on-device verifier: read the merged image back, treat the
    # table area as 0xFF and recompute every page CRC.
    segments = [Segment(0x08000000, bytes(range(32)))]
    result = inject_page_crc(segments, 0x08000000, 0x08000040, 16, 0x08000030)
    assert result.ok
    merged_memory: dict[int, int] = {}
    for seg in result.segments:
        for i, value in enumerate(seg.data):
            merged_memory[seg.start + i] = value
    table = (result.table_address, result.table_end)
    for page in result.pages:
        data = materialize(merged_memory, page.start, page.end, table)
        assert crc32hex(data) == page.crc32
    # the table read back from the image carries the same CRCs little-endian
    for index, page in enumerate(result.pages):
        addr = result.table_address + index * TABLE_ENTRY_SIZE
        stored = int.from_bytes(bytes(merged_memory[addr + i] for i in range(4)), "little")
        assert stored == int(page.crc32, 16)


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

def test_api_happy_path_returns_crcs_table_segments_and_normalized_hex():
    text = "\n".join([
        linear_base(0x0800),
        data_rec(0x0000, b"\x01\x02\x03\x04\x05\x06\x07\x08"),
        linear_entry(0x080001C1),
        EOF,
    ])
    resp = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0x08000000,
            "range_end": 0x08000020,
            "page_size": 16,
            "table_address": 0x08000018,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["hex_ok"] is True
    assert body["params_valid"] is True
    assert body["error_count"] == 0
    assert body["entry_address"] == 0x080001C1
    assert body["total_bytes"] == 8
    assert body["page_count"] == 2 and body["max_pages"] == MAX_PAGES
    assert body["table_address"] == 0x08000018
    assert body["table_size"] == 8 and body["table_end"] == 0x08000020
    assert body["param_errors"] == []

    pages = body["pages"]
    assert [p["index"] for p in pages] == [0, 1]
    assert pages[0]["start"] == 0x08000000 and pages[0]["end"] == 0x08000010
    assert pages[0]["present_bytes"] == 8 and pages[0]["table_bytes"] == 0
    expected_crc0 = zlib.crc32(b"\x01\x02\x03\x04\x05\x06\x07\x08" + b"\xff" * 8) & 0xFFFFFFFF
    assert pages[0]["crc32"] == f"{expected_crc0:08x}"
    expected_crc1 = zlib.crc32(b"\xff" * 16) & 0xFFFFFFFF
    assert pages[1]["crc32"] == f"{expected_crc1:08x}"
    assert pages[1]["table_bytes"] == 8

    table = base64.b64decode(body["table_base64"])
    assert table == expected_crc0.to_bytes(4, "little") + expected_crc1.to_bytes(4, "little")

    # final segment summary: original data plus the merged table segment
    assert body["final_total_bytes"] == 8 + 8
    segments = body["segments"]
    assert [(s["start"], s["end"], s["size"]) for s in segments] == [
        (0x08000000, 0x08000008, 8),
        (0x08000018, 0x08000020, 8),
    ]
    assert base64.b64decode(segments[1]["data_base64"]) == table

    # the exported HEX is normalized, preserves the entry and round-trips
    exported = body["exported_hex"]
    assert exported is not None and exported.endswith(EOF + "\n")
    assert ":04000005080001C12D" in exported  # entry preserved as a type 05 record
    reparsed = parse_intel_hex(exported)
    assert reparsed.ok and reparsed.entry_address == 0x080001C1
    assert reparsed.total_bytes == body["final_total_bytes"]
    assert [(s.start, s.data) for s in reparsed.segments] == [
        (s["start"], base64.b64decode(s["data_base64"])) for s in segments
    ]
    # every data record carries at most 16 bytes
    for line in exported.splitlines():
        if line.startswith(":") and line[7:9] == "00":
            assert int(line[1:3], 16) <= 16


def test_api_invalid_hex_computes_nothing():
    resp = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": ":00000001FE\n",  # bad checksum, no EOF
            "range_start": 0,
            "range_end": 16,
            "page_size": 8,
            "table_address": 8,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["hex_ok"] is False
    assert body["params_valid"] is None
    assert body["pages"] == [] and body["segments"] == []
    assert body["table_base64"] is None and body["exported_hex"] is None
    assert body["final_total_bytes"] == 0
    assert body["param_errors"] == []
    assert body["error_count"] >= 1


def test_api_param_failure_injects_nothing():
    text = "\n".join([data_rec(0x0008, b"\x01\x02\x03\x04"), EOF])
    body = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0,
            "range_end": 16,
            "page_size": 8,
            "table_address": 8,  # overlaps the explicit bytes at 8..12
        },
    ).json()
    assert body["ok"] is False
    assert body["hex_ok"] is True
    assert body["params_valid"] is False
    assert [e["kind"] for e in body["param_errors"]] == ["table_region_occupied"]
    err = body["param_errors"][0]
    assert err["start"] == 8 and err["end"] == 12 and err["address"] == 8
    assert body["pages"] == [] and body["segments"] == []
    assert body["table_base64"] is None and body["exported_hex"] is None


def test_api_table_alignment_and_range_errors_are_aggregated():
    text = "\n".join([EOF])
    body = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0,
            "range_end": 16,
            "page_size": 8,
            "table_address": 14,  # misaligned and the table would leave the range
        },
    ).json()
    assert body["params_valid"] is False
    assert {e["kind"] for e in body["param_errors"]} == {
        "table_address_unaligned",
        "table_out_of_range",
    }


def test_api_negative_table_address_is_a_param_error_not_422():
    text = "\n".join([EOF])
    resp = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0,
            "range_end": 16,
            "page_size": 8,
            "table_address": -8,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["params_valid"] is False
    assert [e["kind"] for e in body["param_errors"]] == ["table_out_of_range"]


def test_api_too_many_pages():
    text = "\n".join([EOF])
    body = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0,
            "range_end": MAX_PAGES + 1,
            "page_size": 1,
            "table_address": 0,
        },
    ).json()
    assert body["ok"] is False
    assert body["params_valid"] is False
    assert [e["kind"] for e in body["param_errors"]] == ["too_many_pages"]
    assert body["page_count"] == MAX_PAGES + 1
    assert body["exported_hex"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"page_size": 0},
        {"page_size": -1},
        {"range_start": -1},
        {"range_end": -1},
        {"table_address": "0x10"},
        {"hex_text": 123},
    ],
)
def test_api_schema_rejects_bad_types_and_non_positive_page_size(payload):
    request = {
        "hex_text": "\n".join([EOF]),
        "range_start": 0,
        "range_end": 16,
        "page_size": 8,
        "table_address": 8,
    }
    request.update(payload)
    resp = client.post("/api/v1/inject-page-crc", json=request)
    assert resp.status_code == 422


def test_api_result_is_stable_across_record_splitting():
    payload = bytes(range(8))
    one = "\n".join([linear_base(0x0800), data_rec(0x0000, payload), EOF])
    many = "\n".join([
        linear_base(0x0800),
        data_rec(0x0000, payload[:3]),
        data_rec(0x0003, payload[3:]),
        EOF,
    ])
    request = {
        "range_start": 0x08000000,
        "range_end": 0x08000020,
        "page_size": 16,
        "table_address": 0x08000018,
    }
    a = client.post("/api/v1/inject-page-crc", json={**request, "hex_text": one}).json()
    b = client.post("/api/v1/inject-page-crc", json={**request, "hex_text": many}).json()
    assert a["ok"] and b["ok"]
    assert a["exported_hex"] == b["exported_hex"]
    assert [p["crc32"] for p in a["pages"]] == [p["crc32"] for p in b["pages"]]
    assert a["table_base64"] == b["table_base64"]


def test_api_exported_hex_is_a_fixed_point_of_export():
    text = "\n".join([linear_base(0x0800), data_rec(0x0000, bytes(range(8))), EOF])
    body = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0x08000000,
            "range_end": 0x08000020,
            "page_size": 16,
            "table_address": 0x08000018,
        },
    ).json()
    assert body["ok"] is True
    reexported = client.post(
        "/api/v1/export-hex",
        json={"hex_text": body["exported_hex"], "max_data_bytes": 16, "include_entry": True},
    ).json()
    assert reexported["ok"] is True
    assert reexported["exported_hex"] == body["exported_hex"]
