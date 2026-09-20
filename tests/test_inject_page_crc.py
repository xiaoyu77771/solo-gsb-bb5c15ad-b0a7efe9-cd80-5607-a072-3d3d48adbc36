"""Tests for the per-page CRC injection module and /api/v1/inject-page-crc."""

import zlib

import pytest
from fastapi.testclient import TestClient

from app.hex_parser import Segment, parse_intel_hex
from app.page_crc import (
    CRC_ENTRY_SIZE,
    FILLER_BYTE,
    MAX_PAGES,
    inject_page_crc,
    validate_page_crc_params,
)
from app.main import app

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


def materialize(image: dict[int, int], start: int, size: int,
                table_start: int, table_end: int) -> bytes:
    """Independently materialize one page the way the service must."""
    out = bytearray(size)
    for offset in range(size):
        address = start + offset
        if table_start <= address < table_end:
            out[offset] = FILLER_BYTE
        else:
            out[offset] = image.get(address, FILLER_BYTE)
    return bytes(out)


# ---------------------------------------------------------------------------
# parameter validation (pure logic)
# ---------------------------------------------------------------------------

def test_valid_params_have_no_errors():
    errors, pages, table_end = validate_page_crc_params([], 0, 32, 16, 0)
    assert errors == []
    assert pages == 2
    assert table_end == 2 * CRC_ENTRY_SIZE


def test_range_must_be_non_empty_half_open_interval():
    errors, pages, table_end = validate_page_crc_params([], 16, 16, 16, 16)
    assert [e.kind for e in errors] == ["invalid_range"]
    assert pages == 0 and table_end == 16
    errors, _, _ = validate_page_crc_params([], 32, 16, 16, 0)
    assert [e.kind for e in errors] == ["invalid_range"]


def test_range_end_may_reach_end_of_32_bit_space():
    start = 0x100000000 - 16
    errors, pages, _ = validate_page_crc_params([], start, 0x100000000, 16, start)
    assert errors == []
    assert pages == 1


def test_both_endpoints_must_be_page_aligned():
    errors, _, _ = validate_page_crc_params([], 1, 16, 4, 0)
    assert [e.kind for e in errors] == ["range_unaligned"]
    errors, _, _ = validate_page_crc_params([], 0, 18, 4, 0)
    assert [e.kind for e in errors] == ["range_unaligned"]


def test_table_address_must_be_word_aligned():
    errors, _, _ = validate_page_crc_params([], 0, 32, 16, 2)
    assert [e.kind for e in errors] == ["table_address_unaligned"]
    errors, _, _ = validate_page_crc_params([], 0, 32, 16, 13)
    assert [e.kind for e in errors] == ["table_address_unaligned"]
    # a 4-byte aligned table is fine even when page_size itself is not a
    # multiple of 4: 16 pages of 6 bytes -> 64-byte table at 4, inside [0,96)
    errors, _, _ = validate_page_crc_params([], 0, 96, 6, 4)
    assert errors == []


def test_table_must_lie_wholly_inside_the_range():
    # table starts before range
    errors, _, end = validate_page_crc_params([], 16, 48, 16, 12)
    assert [e.kind for e in errors] == ["table_out_of_range"]
    assert end == 12 + 2 * CRC_ENTRY_SIZE
    # table extends past range_end
    errors, _, _ = validate_page_crc_params([], 0, 32, 16, 28)
    assert [e.kind for e in errors] == ["table_out_of_range"]
    # table end touching range_end is allowed
    assert validate_page_crc_params([], 0, 32, 16, 24)[0] == []


def test_table_region_must_have_no_explicit_bytes():
    segments = [Segment(20, b"\x00")]
    errors, _, _ = validate_page_crc_params(segments, 0, 32, 16, 16)
    assert [e.kind for e in errors] == ["table_region_not_empty"]
    assert errors[0].start == 20 and errors[0].end == 21

    # explicit erased bytes (0xFF) still count as occupied
    segments = [Segment(18, b"\xff\xff")]
    errors, _, _ = validate_page_crc_params(segments, 0, 32, 16, 16)
    assert [e.kind for e in errors] == ["table_region_not_empty"]
    assert errors[0].start == 18 and errors[0].end == 20


def test_table_occupancy_is_clipped_to_the_table_region():
    # table at [0,8): byte 0 is inside, byte 30 outside; one run reported
    segments = [Segment(0, b"\x01"), Segment(30, b"\x02")]
    errors, _, _ = validate_page_crc_params(segments, 0, 32, 16, 0)
    assert [e.kind for e in errors] == ["table_region_not_empty"]
    assert errors[0].start == 0 and errors[0].end == 1


def test_explicit_bytes_outside_table_region_are_fine():
    segments = [Segment(8, b"\x01\x02"), Segment(24, b"\x03")]
    assert validate_page_crc_params(segments, 0, 32, 16, 0)[0] == []


def test_page_count_is_capped():
    # page_size 4 so the table (4 bytes/page) exactly fills the range
    assert validate_page_crc_params([], 0, 4 * MAX_PAGES, 4, 0)[0] == []
    errors, _, _ = validate_page_crc_params([], 0, 4 * (MAX_PAGES + 1), 4, 0)
    assert [e.kind for e in errors] == ["too_many_pages"]


def test_unaligned_range_and_unaligned_table_are_reported_together():
    errors, _, _ = validate_page_crc_params([], 1, 16, 4, 2)
    assert {e.kind for e in errors} == {
        "range_unaligned",
        "table_address_unaligned",
    }
    # malformed range suppresses every other check
    errors, _, _ = validate_page_crc_params([], 32, 16, 4, 0)
    assert [e.kind for e in errors] == ["invalid_range"]


def test_unaligned_table_that_also_spills_reports_both():
    errors, _, _ = validate_page_crc_params([], 0, 16, 16, 15)
    assert {e.kind for e in errors} == {
        "table_address_unaligned",
        "table_out_of_range",
    }


# ---------------------------------------------------------------------------
# injection semantics (pure logic)
# ---------------------------------------------------------------------------

def test_missing_bytes_materialize_as_ff():
    # 2 pages of 8 bytes over [0,16); the 8-byte table sits at [8,16),
    # leaving the explicit bytes at [0,2) outside the table region.
    segments = [Segment(0, b"\x01\x02")]
    result = inject_page_crc(segments, 0, 16, 8, 8)
    assert result.ok and result.page_count == 2
    image = {0: 0x01, 1: 0x02}
    table_start, table_end = 8, 16
    expected_0 = zlib.crc32(materialize(image, 0, 8, table_start, table_end)) & 0xFFFFFFFF
    expected_1 = zlib.crc32(materialize(image, 8, 8, table_start, table_end)) & 0xFFFFFFFF
    assert [p.crc32 for p in result.pages] == [expected_0, expected_1]
    assert expected_0 == zlib.crc32(b"\x01\x02" + b"\xff" * 6) & 0xFFFFFFFF
    assert expected_1 == zlib.crc32(b"\xff" * 8) & 0xFFFFFFFF
    assert [p.present_bytes for p in result.pages] == [2, 0]
    assert [p.size for p in result.pages] == [8, 8]


def test_table_bytes_are_little_endian_and_merged_into_image():
    result = inject_page_crc([], 0, 16, 16, 0)
    assert result.page_count == 1
    expected_crc = zlib.crc32(b"\xff" * 16) & 0xFFFFFFFF
    assert result.pages[0].crc32 == expected_crc

    merged = {a: b for s in result.segments for a, b in
              zip(range(s.start, s.end), s.data)}
    word = bytes(merged[k] for k in range(0, 4))
    assert word == expected_crc.to_bytes(4, "little")
    # the 0xFF fill outside the table is not emitted as explicit data
    assert set(merged) == {0, 1, 2, 3}
    assert result.segments == [Segment(0, expected_crc.to_bytes(4, "little"))]


def test_page_overlapping_table_reads_table_as_ff():
    # one page of 8 bytes over [0,8); the table occupies [0,4). Every byte,
    # including the table words, must be checksummed as 0xFF.
    result = inject_page_crc([], 0, 8, 8, 0)
    assert result.page_count == 1
    assert result.pages[0].crc32 == zlib.crc32(b"\xff" * 8) & 0xFFFFFFFF
    assert result.pages[0].present_bytes == 0


def test_image_bytes_inside_table_region_are_rejected():
    result = inject_page_crc([Segment(1, b"\x00")], 0, 16, 16, 0)
    assert not result.ok and not result.params_valid
    assert [e.kind for e in result.param_errors] == ["table_region_not_empty"]
    assert result.pages == [] and result.segments == []


def test_segments_outside_range_are_preserved_and_merged():
    # image byte before the range and image bytes inside a non-table area
    segments = [Segment(0x100, b"\xAA"), Segment(0x08000008, b"\x01\x02")]
    result = inject_page_crc(segments, 0x08000000, 0x08000010, 16, 0x08000000)
    assert result.ok
    merged = {a: b for s in result.segments for a, b in
              zip(range(s.start, s.end), s.data)}
    assert merged[0x100] == 0xAA  # out-of-range byte kept
    assert merged[0x08000008] == 0x01 and merged[0x08000009] == 0x02
    assert merged[0x08000003] is not None  # last table word byte
    # table [0,4) and data [8,10) are separated by missing bytes 4..7 ->
    # the page splits into two segments; plus the byte at 0x100
    assert [s.start for s in result.segments] == [
        0x100,
        0x08000000,
        0x08000008,
    ]


def test_param_failure_produces_nothing():
    result = inject_page_crc([], 3, 16, 4, 0)
    assert not result.ok
    assert result.params_valid is False
    assert result.pages == [] and result.segments == []
    assert [e.kind for e in result.param_errors] == ["range_unaligned"]


def test_crc_is_stable_across_record_splitting():
    one = parse_intel_hex(
        "\n".join([linear_base(0x0800), data_rec(0x0000, bytes(range(8))), EOF])
    )
    many = parse_intel_hex(
        "\n".join([
            linear_base(0x0800),
            data_rec(0x0000, bytes(range(3))),
            data_rec(0x0003, bytes(range(3, 8))),
            EOF,
        ])
    )
    args = (0x08000010, 0x08000020, 16, 0x08000010)
    a = inject_page_crc(one.segments, *args)
    b = inject_page_crc(many.segments, *args)
    assert [p.crc32 for p in a.pages] == [p.crc32 for p in b.pages]
    assert a.segments == b.segments


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

def test_api_happy_path_injects_table_and_exports_hex():
    text = "\n".join([
        linear_base(0x0800),
        data_rec(0x0000, bytes(range(16))),
        EOF,
    ])
    # two pages over [0x08000000, +32); table occupies [+16,+24)
    resp = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0x08000000,
            "range_end": 0x08000020,
            "page_size": 16,
            "table_address": 0x08000010,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["hex_ok"] is True
    assert body["params_valid"] is True
    assert body["page_count"] == 2
    assert body["table_end"] == 0x08000018
    assert body["max_pages"] == MAX_PAGES

    image = {0x08000000 + i: i for i in range(16)}
    table_start, table_end = 0x08000010, 0x08000018
    expected_0 = zlib.crc32(
        materialize(image, 0x08000000, 16, table_start, table_end)
    ) & 0xFFFFFFFF
    expected_1 = zlib.crc32(
        materialize(image, 0x08000010, 16, table_start, table_end)
    ) & 0xFFFFFFFF
    pages = body["pages"]
    assert [p["index"] for p in pages] == [0, 1]
    assert pages[0]["crc32"] == expected_0
    assert pages[1]["crc32"] == expected_1
    assert pages[0]["present_bytes"] == 16
    assert pages[1]["present_bytes"] == 0
    assert pages[1]["size"] == 16

    exported = body["exported_hex"]
    assert exported is not None and exported.endswith("\n")
    # the exported text re-parses and carries the little-endian table words
    reparsed = parse_intel_hex(exported)
    assert reparsed.ok
    flat = {a: b for s in reparsed.segments for a, b in
            zip(range(s.start, s.end), s.data)}
    for index, crc in enumerate((expected_0, expected_1)):
        word = bytes(flat[table_start + 4 * index + k] for k in range(4))
        assert word == crc.to_bytes(4, "little")

    # final segment summaries describe the image with the table merged in
    assert body["segment_count"] == len(body["segments"])
    assert body["injected_total_bytes"] == sum(s["size"] for s in body["segments"])
    assert body["injected_total_bytes"] == 16 + 8
    assert body["record_count"] >= 1 and body["data_record_count"] >= 1


def test_api_table_inside_first_page_crc_sees_ff_there():
    text = "\n".join([linear_base(0x0800), EOF])
    body = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0x08000000,
            "range_end": 0x08000020,
            "page_size": 16,
            "table_address": 0x08000000,
        },
    ).json()
    assert body["ok"] is True
    # page 0 geometrically contains the table [+0,+8); those bytes read 0xFF
    assert body["pages"][0]["crc32"] == zlib.crc32(b"\xff" * 16) & 0xFFFFFFFF
    assert body["pages"][1]["crc32"] == zlib.crc32(b"\xff" * 16) & 0xFFFFFFFF


def test_api_invalid_hex_does_not_validate_params_or_inject():
    resp = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": ":00000001FE\n",
            "range_start": 0,
            "range_end": 16,
            "page_size": 16,
            "table_address": 0,
        },
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["hex_ok"] is False
    assert body["params_valid"] is None
    assert body["pages"] == []
    assert body["exported_hex"] is None
    assert body["segments"] == []
    assert body["param_errors"] == []
    assert body["page_count"] == 0 and body["record_count"] == 0
    assert body["error_count"] >= 1


def test_api_unaligned_range_and_table():
    text = "\n".join([EOF])
    body = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 1,
            "range_end": 16,
            "page_size": 4,
            "table_address": 15,
        },
    ).json()
    assert body["ok"] is False and body["params_valid"] is False
    assert body["pages"] == [] and body["exported_hex"] is None
    assert body["segments"] == []
    assert body["table_end"] == 15  # grid not established, table_address echoed
    assert {e["kind"] for e in body["param_errors"]} == {
        "range_unaligned",
        "table_address_unaligned",
    }


def test_api_table_out_of_range():
    text = "\n".join([EOF])
    body = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0,
            "range_end": 16,
            "page_size": 16,
            "table_address": 16,  # table [16,20) spills out of [0,16)
        },
    ).json()
    assert [e["kind"] for e in body["param_errors"]] == ["table_out_of_range"]
    err = body["param_errors"][0]
    assert err["start"] == 16 and err["end"] == 20


def test_api_table_region_not_empty_rejects_explicit_ff_too():
    # explicit 0xFF byte at address 2, inside the 4-byte table [0,4)
    text = "\n".join([data_rec(0x0002, b"\xff"), EOF])
    body = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0,
            "range_end": 16,
            "page_size": 16,
            "table_address": 0,
        },
    ).json()
    assert body["ok"] is False
    assert [e["kind"] for e in body["param_errors"]] == ["table_region_not_empty"]
    err = body["param_errors"][0]
    assert err["start"] == 2 and err["end"] == 3


def test_api_too_many_pages():
    text = "\n".join([EOF])
    body = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0,
            "range_end": 4 * (MAX_PAGES + 1),
            "page_size": 4,
            "table_address": 0,
        },
    ).json()
    assert body["ok"] is False
    assert [e["kind"] for e in body["param_errors"]] == ["too_many_pages"]
    assert body["pages"] == []


def test_api_entry_address_is_preserved_in_export():
    text = "\n".join([
        linear_base(0x0800),
        data_rec(0x0000, b"\x01"),
        rec(0x05, 0, (0x080001C1).to_bytes(4, "big")),
        EOF,
    ])
    body = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0x08000000,
            "range_end": 0x08000020,
            "page_size": 16,
            "table_address": 0x08000010,
        },
    ).json()
    assert body["entry_address"] == 0x080001C1
    assert body["entry_record_count"] == 1
    assert rec(0x05, 0, (0x080001C1).to_bytes(4, "big")) in body["exported_hex"]


def test_api_include_entry_false_omits_entry_record():
    text = "\n".join([
        rec(0x05, 0, (0x080001C1).to_bytes(4, "big")),
        EOF,
    ])
    body = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0x08000000,
            "range_end": 0x08000010,
            "page_size": 16,
            "table_address": 0x08000000,
            "include_entry": False,
        },
    ).json()
    assert body["ok"] is True
    assert body["entry_record_count"] == 0
    assert ":04000005080001C12D" not in body["exported_hex"]


def test_api_export_is_normalized_fixed_point():
    # image occupies only [+0,+8); 3 pages over 48 bytes; the 12-byte
    # table at [+32,+44) lands in blank erased area
    payload = bytes(range(8))
    text = "\n".join([linear_base(0x0800), data_rec(0x0000, payload), EOF])
    body = client.post(
        "/api/v1/inject-page-crc",
        json={
            "hex_text": text,
            "range_start": 0x08000000,
            "range_end": 0x08000030,
            "page_size": 16,
            "table_address": 0x08000020,
            "max_data_bytes": 7,
        },
    ).json()
    assert body["ok"] is True
    exported = body["exported_hex"]
    # re-exporting the injected image reproduces the text byte for byte
    second = client.post(
        "/api/v1/export-hex",
        json={"hex_text": exported, "max_data_bytes": 7, "include_entry": True},
    ).json()
    assert second["exported_hex"] == exported


@pytest.mark.parametrize(
    "payload",
    [
        {"page_size": 0},
        {"range_start": -1},
        {"range_end": -1},
        {"table_address": -1},
        {"max_data_bytes": 0},
        {"max_data_bytes": 256},
        {"hex_text": 123},
    ],
)
def test_api_schema_rejects_bad_parameters(payload):
    request = {
        "hex_text": "\n".join([EOF]),
        "range_start": 0,
        "range_end": 16,
        "page_size": 16,
        "table_address": 0,
    }
    request.update(payload)
    resp = client.post("/api/v1/inject-page-crc", json=request)
    assert resp.status_code == 422
