"""Per-page IEEE CRC-32 table injection over an aligned address range.

The parser turns Intel HEX text into sparse, address-ordered segments.
This module lays a fixed grid of equal-sized pages over an explicit
half-open address range ``[range_start, range_end)`` whose endpoints are
aligned to ``page_size``, materializes every page to exactly
``page_size`` bytes (addresses absent from the image read as ``0xFF``,
the NOR-flash erased value) and computes one IEEE CRC-32 per page — the
reflected ``0xEDB88320`` polynomial checksum with init/xorout
``0xFFFFFFFF`` exposed as ``zlib.crc32``.

The resulting table of ``page_count`` little-endian 32-bit entries is
meant to live inside the very range it protects, at ``table_address``:

* ``table_address`` must be 4-byte aligned and the whole table
  ``[table_address, table_address + 4 * page_count)`` must lie inside
  ``[range_start, range_end)``;
* the table area must be unoccupied in the source image: any explicit
  image byte under the table is reported and nothing is injected;
* while the CRC of *every* page is computed, the whole table area is
  treated as ``0xFF`` (its erased state), so the injected table never
  feeds back into the checksums.

On success the little-endian table is merged into the image and the
final contiguous segments are returned together with the per-page CRCs;
the route additionally re-exports the merged image as normalized Intel
HEX. At most ``MAX_PAGES`` (4096) pages are permitted.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from .hex_parser import Segment

ADDR_SPAN = 1 << 32  # exclusive upper bound of the 32-bit address space

FILLER_BYTE = 0xFF
TABLE_ENTRY_SIZE = 4  # one little-endian CRC-32 per page
MAX_PAGES = 4096


@dataclass
class CrcParamError:
    """A parameter-level problem that prevents the table from being injected.

    ``kind`` is one of ``invalid_range``, ``range_unaligned``,
    ``too_many_pages``, ``table_address_unaligned``, ``table_out_of_range``
    or ``table_region_occupied``.
    """

    kind: str
    message: str
    start: int | None = None
    end: int | None = None
    address: int | None = None


@dataclass
class PageCrc:
    """Per-page summary; the materialized page bytes are not retained."""

    index: int
    start: int
    end: int  # exclusive
    size: int  # always page_size
    present_bytes: int  # bytes explicitly supplied by the image (rest are 0xFF)
    table_bytes: int  # bytes of this page covered by the CRC table (read as 0xFF)
    crc32: str  # IEEE CRC-32 of the materialized page, 8 lowercase hex digits


@dataclass
class PageCrcResult:
    """The injection outcome; pages/table/segments are empty on failure."""

    ok: bool
    params_valid: bool
    param_errors: list[CrcParamError]
    range_start: int
    range_end: int
    page_size: int
    page_count: int
    table_address: int
    table_size: int  # page_count * TABLE_ENTRY_SIZE
    pages: list[PageCrc]
    table: bytes  # little-endian CRC table merged into the image
    segments: list[Segment]  # final image with the table merged in
    final_total_bytes: int

    @property
    def table_end(self) -> int:
        """Exclusive end of the CRC table."""
        return self.table_address + self.table_size


def _fmt(addr: int) -> str:
    return f"0x{addr:08X}"


def _to_map(segments: list[Segment]) -> dict[int, int]:
    """Flatten assembled segments into an ``address -> byte`` mapping."""
    memory: dict[int, int] = {}
    for seg in segments:
        for i, value in enumerate(seg.data):
            memory[seg.start + i] = value
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
    for addr in ordered[1:]:
        if addr == prev + 1:
            blob.append(memory[addr])
        else:
            segments.append(Segment(start=start, data=bytes(blob)))
            start = addr
            blob = bytearray([memory[addr]])
        prev = addr
    segments.append(Segment(start=start, data=bytes(blob)))
    return segments


def _contiguous_runs(addresses: list[int]) -> list[tuple[int, int]]:
    """Group sorted addresses into maximal ``(start, end)`` exclusive runs."""
    runs: list[list[int]] = []
    for addr in addresses:
        if runs and addr == runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([addr, addr + 1])
    return [(start, end) for start, end in runs]


def _validate_params(
    memory: dict[int, int],
    range_start: int,
    range_end: int,
    page_size: int,
    table_address: int,
) -> list[CrcParamError]:
    """Validate the range, table placement and table-area occupancy.

    Returns every error found; an empty list means the CRC table may be
    computed and injected. ``page_size`` is expected to be a positive
    integer already (enforced at the request schema level). Range shape
    problems suppress the alignment/page-count checks, and an unsound
    range suppresses the table placement/occupancy checks (the table
    size derives from the page count); the table-address alignment is
    always checked.
    """
    errors: list[CrcParamError] = []

    range_ok = 0 <= range_start < range_end <= ADDR_SPAN
    if not range_ok:
        if range_start < 0 or range_end < 0:
            message = (
                f"range endpoints must be non-negative: "
                f"[{_fmt(range_start)}, {_fmt(range_end)})"
            )
        elif range_end > ADDR_SPAN:
            message = (
                f"range exceeds the 32-bit address space: "
                f"[{_fmt(range_start)}, {_fmt(range_end)}), end must be <= 0x100000000"
            )
        else:
            message = (
                f"range must be a non-empty half-open interval with "
                f"range_start < range_end: [{_fmt(range_start)}, {_fmt(range_end)})"
            )
        errors.append(
            CrcParamError(
                kind="invalid_range", message=message, start=range_start, end=range_end
            )
        )

    aligned = False
    page_count = 0
    if range_ok and page_size > 0:
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
                CrcParamError(
                    kind="range_unaligned",
                    message=(
                        "both range endpoints must be aligned to page_size="
                        f"{page_size}: " + "; ".join(detail)
                    ),
                    start=range_start,
                    end=range_end,
                )
            )

    if aligned:
        page_count = (range_end - range_start) // page_size
        if page_count > MAX_PAGES:
            errors.append(
                CrcParamError(
                    kind="too_many_pages",
                    message=(
                        f"range requires {page_count} page(s) of {page_size} byte(s); "
                        f"at most {MAX_PAGES} pages are allowed "
                        f"(widen page_size or shrink the range)"
                    ),
                    start=range_start,
                    end=range_end,
                )
            )

    if table_address % TABLE_ENTRY_SIZE:
        errors.append(
            CrcParamError(
                kind="table_address_unaligned",
                message=(
                    f"table_address {_fmt(table_address)} must be aligned to "
                    f"{TABLE_ENTRY_SIZE} bytes (one little-endian CRC-32 per page)"
                ),
                address=table_address,
            )
        )

    if aligned and 0 < page_count <= MAX_PAGES:
        table_size = page_count * TABLE_ENTRY_SIZE
        table_end = table_address + table_size
        if not (range_start <= table_address and table_end <= range_end):
            errors.append(
                CrcParamError(
                    kind="table_out_of_range",
                    message=(
                        f"CRC table [{_fmt(table_address)}, {_fmt(table_end)}) "
                        f"({table_size} byte(s) for {page_count} page(s)) must lie "
                        f"inside the page range [{_fmt(range_start)}, {_fmt(range_end)})"
                    ),
                    start=table_address,
                    end=table_end,
                    address=table_address,
                )
            )
        else:
            occupied = sorted(a for a in memory if table_address <= a < table_end)
            for run_start, run_end in _contiguous_runs(occupied):
                errors.append(
                    CrcParamError(
                        kind="table_region_occupied",
                        message=(
                            f"CRC table area [{_fmt(table_address)}, {_fmt(table_end)}) "
                            f"overlaps {run_end - run_start} explicit image byte(s) at "
                            f"[{_fmt(run_start)}, {_fmt(run_end)}); the table area must "
                            f"be unprogrammed (0xFF)"
                        ),
                        start=run_start,
                        end=run_end,
                        address=table_address,
                    )
                )
    return errors


def validate_page_crc_params(
    segments: list[Segment],
    range_start: int,
    range_end: int,
    page_size: int,
    table_address: int,
) -> list[CrcParamError]:
    """Validate the range, table placement and table-area occupancy.

    Returns every error found; an empty list means the CRC table may be
    computed and injected.
    """
    return _validate_params(
        _to_map(segments), range_start, range_end, page_size, table_address
    )


def inject_page_crc(
    segments: list[Segment],
    range_start: int,
    range_end: int,
    page_size: int,
    table_address: int,
) -> PageCrcResult:
    """Validate parameters, compute per-page CRC-32s and inject the table.

    On any parameter error nothing is injected: ``pages``/``segments``
    are empty and ``table`` is null-length.
    """
    memory = _to_map(segments)
    errors = _validate_params(memory, range_start, range_end, page_size, table_address)
    page_count = (
        (range_end - range_start) // page_size
        if range_start >= 0 and range_end >= range_start and page_size > 0
        else 0
    )
    table_size = page_count * TABLE_ENTRY_SIZE
    if errors:
        return PageCrcResult(
            ok=False,
            params_valid=False,
            param_errors=errors,
            range_start=range_start,
            range_end=range_end,
            page_size=page_size,
            page_count=page_count,
            table_address=table_address,
            table_size=table_size,
            pages=[],
            table=b"",
            segments=[],
            final_total_bytes=0,
        )

    table_end = table_address + table_size
    pages: list[PageCrc] = []
    crcs: list[int] = []
    for index in range(page_count):
        start = range_start + index * page_size
        end = start + page_size
        # The whole table area reads as 0xFF in every page CRC, exactly
        # as if the table had not been injected yet.
        data = bytes(
            FILLER_BYTE if table_address <= addr < table_end else memory.get(addr, FILLER_BYTE)
            for addr in range(start, end)
        )
        crc = zlib.crc32(data) & 0xFFFFFFFF
        crcs.append(crc)
        pages.append(
            PageCrc(
                index=index,
                start=start,
                end=end,
                size=page_size,
                present_bytes=sum(1 for addr in range(start, end) if addr in memory),
                table_bytes=max(0, min(end, table_end) - max(start, table_address)),
                crc32=f"{crc:08x}",
            )
        )

    table = b"".join(crc.to_bytes(TABLE_ENTRY_SIZE, "little") for crc in crcs)
    merged = dict(memory)
    for offset, value in enumerate(table):
        merged[table_address + offset] = value
    merged_segments = _build_segments(merged)

    return PageCrcResult(
        ok=True,
        params_valid=True,
        param_errors=[],
        range_start=range_start,
        range_end=range_end,
        page_size=page_size,
        page_count=page_count,
        table_address=table_address,
        table_size=table_size,
        pages=pages,
        table=table,
        segments=merged_segments,
        final_total_bytes=len(merged),
    )
