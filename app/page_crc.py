"""Per-page IEEE CRC-32 table injection over a page-aligned address range.

The parser turns Intel HEX text into sparse, address-ordered segments.
This module lays a fixed grid of equal-sized pages over an explicit
half-open address range ``[range_start, range_end)`` aligned to
``page_size``, materializes every page to exactly ``page_size`` bytes
(addresses absent from the image are filled with ``0xFF``, the NOR-flash
erased value), and computes one IEEE CRC-32 (the standard
``zlib.crc32``: reflected polynomial 0xEDB88320, init/final-XOR
0xFFFFFFFF) per page.

The CRC table holds one little-endian 32-bit entry per page, in page
order. While every page CRC is computed the *whole table region* is
treated as ``0xFF`` -- including the bytes of other entries -- so a page
that geometrically overlaps the table sees erased bytes there. On
success the little-endian table bytes are overlaid onto the image and
the merged, contiguous segments are returned for normalized HEX export.

Safety/placement rules, any violation of which aborts the injection
without producing a result:

* the range must be a non-empty half-open interval inside the 32-bit
  address space, both endpoints aligned to ``page_size``;
* at most ``MAX_PAGES`` pages may be materialized;
* ``table_address`` must be 4-byte aligned (entries are 4-byte words);
* the whole table ``[table_address, table_address + 4*page_count)`` must
  lie inside the range;
* the table region must be blank: it may not carry a single byte
  explicitly supplied by the input image (an explicitly written
  ``0xFF`` counts as occupied -- only genuinely missing bytes are
  accepted), so the injection never silently overwrites image data.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from .hex_parser import Segment

ADDR_SPAN = 1 << 32  # exclusive upper bound of the 32-bit address space

FILLER_BYTE = 0xFF
CRC_ENTRY_SIZE = 4  # one 32-bit little-endian CRC word per page
CRC_ALIGNMENT = 4  # entries are 32-bit words
MAX_PAGES = 1 << 20  # hard cap on materialized pages (1,048,576)


@dataclass
class PageCrcParamError:
    """A parameter-level problem that prevents the injection.

    ``kind`` is one of ``invalid_range``, ``range_unaligned``,
    ``table_address_unaligned``, ``table_out_of_range``,
    ``table_region_not_empty`` or ``too_many_pages``.
    """

    kind: str
    message: str
    start: int | None = None
    end: int | None = None
    address: int | None = None


@dataclass
class PageCrc:
    """Per-page CRC summary; the materialized page bytes are not retained."""

    index: int
    start: int
    end: int  # exclusive
    size: int  # always page_size
    present_bytes: int  # bytes explicitly supplied by the image (rest are 0xFF)
    crc32: int  # unsigned 32-bit IEEE CRC-32


@dataclass
class InjectedImage:
    """Result of the injection; pages/segments are empty on failure."""

    ok: bool
    params_valid: bool
    param_errors: list[PageCrcParamError]
    range_start: int
    range_end: int
    page_size: int
    page_count: int
    table_address: int
    table_end: int  # exclusive; == table_address when it could not be computed
    max_pages: int
    pages: list[PageCrc]
    segments: list[Segment]


def _fmt(addr: int) -> str:
    return f"0x{addr:08X}"


def _flatten(segments: list[Segment]) -> dict[int, int]:
    """Flatten every segment into an address -> byte map."""
    memory: dict[int, int] = {}
    for segment in sorted(segments, key=lambda s: s.start):
        for offset, value in enumerate(segment.data):
            memory[segment.start + offset] = value
    return memory


def _build_segments(memory: dict[int, int]) -> list[Segment]:
    """Merge sparse bytes into contiguous, address-ordered segments."""
    segments: list[Segment] = []
    if not memory:
        return segments
    ordered = sorted(memory)
    start = ordered[0]
    blob = bytearray([memory[start]])
    prev = start
    for address in ordered[1:]:
        if address == prev + 1:
            blob.append(memory[address])
        else:
            segments.append(Segment(start=start, data=bytes(blob)))
            start = address
            blob = bytearray([memory[address]])
        prev = address
    segments.append(Segment(start=start, data=bytes(blob)))
    return segments


def _table_occupancy(
    segments: list[Segment], table_start: int, table_end: int
) -> list[tuple[int, int]]:
    """Find maximal explicit-image byte runs inside the table region."""
    runs: list[tuple[int, int]] = []
    for segment in sorted(segments, key=lambda s: s.start):
        start = max(segment.start, table_start)
        end = min(segment.end, table_end)
        if start < end:
            runs.append((start, end))
    return runs


def validate_page_crc_params(
    segments: list[Segment],
    range_start: int,
    range_end: int,
    page_size: int,
    table_address: int,
) -> tuple[list[PageCrcParamError], int, int]:
    """Validate range/grid/table placement.

    Returns ``(errors, page_count, table_end)``; ``page_count`` is 0 and
    ``table_end`` equals ``table_address`` when the aligned grid cannot be
    established. ``page_size`` is expected to be positive (enforced at
    the request schema level).
    """
    errors: list[PageCrcParamError] = []

    range_ok = 0 <= range_start < range_end <= ADDR_SPAN
    if not range_ok:
        if range_start < 0 or range_end < 0:
            message = (
                "range endpoints must be non-negative: "
                f"[{_fmt(range_start)}, {_fmt(range_end)})"
            )
        elif range_end > ADDR_SPAN:
            message = (
                "range exceeds the 32-bit address space: "
                f"[{_fmt(range_start)}, {_fmt(range_end)}), end must be <= 0x100000000"
            )
        else:
            message = (
                "range must be a non-empty half-open interval with "
                f"range_start < range_end: [{_fmt(range_start)}, {_fmt(range_end)})"
            )
        # A malformed range makes every other placement check meaningless.
        return (
            [PageCrcParamError(kind="invalid_range", message=message,
                               start=range_start, end=range_end)],
            0,
            table_address,
        )

    aligned = range_start % page_size == 0 and range_end % page_size == 0
    if not aligned:
        detail = []
        bad_start = range_start % page_size
        bad_end = range_end % page_size
        if bad_start:
            detail.append(
                f"range_start {_fmt(range_start)} is {bad_start} byte(s) past a "
                f"page_size={page_size} boundary"
            )
        if bad_end:
            detail.append(
                f"range_end {_fmt(range_end)} is {bad_end} byte(s) past a "
                f"page_size={page_size} boundary"
            )
        errors.append(
            PageCrcParamError(
                kind="range_unaligned",
                message=(
                    "both range endpoints must be aligned to page_size="
                    f"{page_size}: " + "; ".join(detail)
                ),
                start=range_start,
                end=range_end,
            )
        )

    if table_address % CRC_ALIGNMENT:
        remainder = table_address % CRC_ALIGNMENT
        errors.append(
            PageCrcParamError(
                kind="table_address_unaligned",
                message=(
                    f"table_address {_fmt(table_address)} must be {CRC_ALIGNMENT}-byte "
                    f"aligned for little-endian 32-bit CRC entries; it is {remainder} "
                    "byte(s) past a word boundary"
                ),
                address=table_address,
            )
        )

    if not aligned:
        return errors, 0, table_address

    page_count = (range_end - range_start) // page_size
    table_end = table_address + page_count * CRC_ENTRY_SIZE

    if page_count > MAX_PAGES:
        errors.append(
            PageCrcParamError(
                kind="too_many_pages",
                message=(
                    f"range requires {page_count} page(s) of {page_size} byte(s); "
                    f"at most {MAX_PAGES} pages are allowed "
                    "(widen page_size or shrink the range)"
                ),
                start=range_start,
                end=range_end,
            )
        )

    table_inside = range_start <= table_address and table_end <= range_end
    if not table_inside:
        errors.append(
            PageCrcParamError(
                kind="table_out_of_range",
                message=(
                    f"CRC table [{_fmt(table_address)}, {_fmt(table_end)}) "
                    f"({page_count * CRC_ENTRY_SIZE} byte(s) for {page_count} page(s)) "
                    f"must lie wholly inside the range "
                    f"[{_fmt(range_start)}, {_fmt(range_end)})"
                ),
                start=table_address,
                end=table_end,
            )
        )
    elif page_count <= MAX_PAGES:
        # Only a small, fully contained table is inspected byte by byte.
        for run_start, run_end in _table_occupancy(segments, table_address, table_end):
            errors.append(
                PageCrcParamError(
                    kind="table_region_not_empty",
                    message=(
                        f"table region [{_fmt(run_start)}, {_fmt(run_end)}) already "
                        f"carries {run_end - run_start} explicit image byte(s); the "
                        "CRC area must be blank (missing bytes, materialized as 0xFF)"
                    ),
                    start=run_start,
                    end=run_end,
                    address=run_start,
                )
            )

    return errors, page_count, table_end


def inject_page_crc(
    segments: list[Segment],
    range_start: int,
    range_end: int,
    page_size: int,
    table_address: int,
) -> InjectedImage:
    """Validate placement, compute per-page CRCs and overlay the table.

    On any parameter error no CRC is computed and no image is produced:
    ``pages`` is empty and ``segments`` is empty.
    """
    errors, page_count, table_end = validate_page_crc_params(
        segments, range_start, range_end, page_size, table_address
    )
    if errors:
        return InjectedImage(
            ok=False,
            params_valid=False,
            param_errors=errors,
            range_start=range_start,
            range_end=range_end,
            page_size=page_size,
            page_count=page_count,
            table_address=table_address,
            table_end=table_end,
            max_pages=MAX_PAGES,
            pages=[],
            segments=[],
        )

    memory = _flatten(segments)
    table_size = page_count * CRC_ENTRY_SIZE

    pages: list[PageCrc] = []
    crc_words = bytearray()
    for index in range(page_count):
        page_start = range_start + index * page_size
        page_end = page_start + page_size
        data = bytearray(page_size)
        present = 0
        for offset, address in enumerate(range(page_start, page_end)):
            if table_address <= address < table_end:
                # The table region reads as erased while *any* page CRC is
                # computed, including for pages that geometrically overlap it.
                data[offset] = FILLER_BYTE
            else:
                value = memory.get(address, FILLER_BYTE)
                data[offset] = value
                if address in memory:
                    present += 1
        crc = zlib.crc32(bytes(data)) & 0xFFFFFFFF
        crc_words.extend(crc.to_bytes(CRC_ENTRY_SIZE, "little"))
        pages.append(
            PageCrc(
                index=index,
                start=page_start,
                end=page_end,
                size=page_size,
                present_bytes=present,
                crc32=crc,
            )
        )

    # Merge the little-endian CRC table into the assembled image.
    assert len(crc_words) == table_size
    for offset, value in enumerate(crc_words):
        memory[table_address + offset] = value

    return InjectedImage(
        ok=True,
        params_valid=True,
        param_errors=[],
        range_start=range_start,
        range_end=range_end,
        page_size=page_size,
        page_count=page_count,
        table_address=table_address,
        table_end=table_end,
        max_pages=MAX_PAGES,
        pages=pages,
        segments=_build_segments(memory),
    )
