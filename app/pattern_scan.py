"""Byte-pattern scanning over assembled firmware images.

A pattern is a whitespace-separated token list. Each token is either an
exact byte (two hex digits, case-insensitive) or the single-byte
wildcard ``??``, optionally followed by a repeat count ``{n}``
(``n >= 1``) that repeats the preceding byte/wildcard. Every pattern
compiles to a value/mask byte pair: exact bytes use mask ``0xFF``,
wildcards mask ``0x00``.

Scanning runs inside the contiguous firmware segments only, so a match
never crosses an inter-segment gap, and overlapping matches of the same
pattern are all reported. Per pattern, an optional half-open address
range restricts the hit *start* addresses, an optional alignment
requires the hit start address to be a multiple of it, and an optional
expected count is compared against the number of hits
(``count_mismatch`` when different; otherwise ``absent`` / ``unique`` /
``ambiguous`` by hit count).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .hex_parser import Segment

ADDR_SPAN = 1 << 32  # exclusive upper bound of the 32-bit address space
MAX_PATTERN_BYTES = 4096  # cap on the compiled value/mask length

ABSENT = "absent"
UNIQUE = "unique"
AMBIGUOUS = "ambiguous"
COUNT_MISMATCH = "count_mismatch"

_TOKEN_RE = re.compile(r"([0-9A-Fa-f]{2}|\?\?)(?:\{([0-9]+)\})?")


@dataclass
class PatternError:
    """A single pattern-level problem; any of them suppresses the scan."""

    kind: str  # invalid_pattern_syntax | pattern_too_long | invalid_range |
    #            invalid_alignment | invalid_expected_count
    message: str
    name: str | None = None
    index: int | None = None


@dataclass
class PatternSpec:
    """Unvalidated per-pattern request fields."""

    name: str
    index: int
    source: str
    range_start: int | None = None
    range_end: int | None = None
    alignment: int = 1
    expected_count: int | None = None


@dataclass
class CompiledPattern:
    """A validated pattern compiled to a value/mask byte pair."""

    name: str
    index: int
    source: str
    values: bytes  # expected bytes, pre-masked (wildcard positions are 0x00)
    masks: bytes  # 0xFF exact byte, 0x00 wildcard
    range_start: int | None
    range_end: int | None
    alignment: int
    expected_count: int | None

    @property
    def length(self) -> int:
        return len(self.values)


@dataclass
class PatternMatch:
    """One hit: the match lies wholly inside a single segment."""

    pattern_index: int
    address: int
    size: int
    data: bytes

    @property
    def end(self) -> int:
        """Exclusive end address."""
        return self.address + self.size


@dataclass
class PatternReport:
    """Per-pattern outcome in request order."""

    name: str
    index: int
    source: str
    status: str  # absent | unique | ambiguous | count_mismatch
    hit_count: int
    expected_count: int | None
    compiled_length: int
    range_start: int | None
    range_end: int | None
    alignment: int
    match_addresses: list[int] = field(default_factory=list)


@dataclass
class ScanResult:
    ok: bool  # True when no pattern reported count_mismatch
    reports: list[PatternReport]
    matches: list[PatternMatch]  # sorted by (address, pattern index)


def _fmt(addr: int) -> str:
    return f"0x{addr:04X}"


def compile_pattern(spec: PatternSpec) -> tuple[CompiledPattern | None, list[PatternError]]:
    """Validate one pattern spec and compile it to a value/mask pair.

    Syntax, length, range, alignment and expected-count problems are
    independent and all collected; the compiled pattern is returned only
    when no problem was found.
    """
    errors: list[PatternError] = []

    def err(kind: str, message: str) -> None:
        errors.append(
            PatternError(kind=kind, message=message, name=spec.name, index=spec.index)
        )

    values = bytearray()
    masks = bytearray()
    tokens = spec.source.split()
    if not tokens:
        err(
            "invalid_pattern_syntax",
            f"pattern '{spec.name}': pattern must contain at least one token",
        )
    for token in tokens:
        match = _TOKEN_RE.fullmatch(token)
        if match is None:
            err(
                "invalid_pattern_syntax",
                f"pattern '{spec.name}': invalid token '{token}': expected two hex "
                "digits or '??', optionally followed by a '{{n}}' repeat count",
            )
            continue
        text, repeat = match.group(1), match.group(2)
        count = int(repeat) if repeat is not None else 1
        if count < 1:
            err(
                "invalid_pattern_syntax",
                f"pattern '{spec.name}': repeat count in token '{token}' must be "
                "a positive integer",
            )
            continue
        if len(values) + count > MAX_PATTERN_BYTES:
            err(
                "pattern_too_long",
                f"pattern '{spec.name}': compiled length would exceed "
                f"{MAX_PATTERN_BYTES} bytes (token '{token}')",
            )
            break
        if text == "??":
            values.extend(b"\x00" * count)
            masks.extend(b"\x00" * count)
        else:
            values.extend(bytes([int(text, 16)]) * count)
            masks.extend(b"\xFF" * count)

    if (spec.range_start is None) != (spec.range_end is None):
        err(
            "invalid_range",
            f"pattern '{spec.name}': range_start and range_end must be given together",
        )
    elif spec.range_start is not None and spec.range_end is not None:
        lo, hi = spec.range_start, spec.range_end
        if lo < 0 or hi < 0 or lo >= hi or hi > ADDR_SPAN:
            err(
                "invalid_range",
                f"pattern '{spec.name}': range must be a non-empty half-open interval "
                f"inside the 32-bit address space, got [{_fmt(lo)}, {_fmt(hi)})",
            )

    if spec.alignment < 1:
        err(
            "invalid_alignment",
            f"pattern '{spec.name}': alignment must be a positive integer, "
            f"got {spec.alignment}",
        )

    if spec.expected_count is not None and spec.expected_count < 0:
        err(
            "invalid_expected_count",
            f"pattern '{spec.name}': expected_count must be >= 0, "
            f"got {spec.expected_count}",
        )

    if errors:
        return None, errors
    return (
        CompiledPattern(
            name=spec.name,
            index=spec.index,
            source=spec.source,
            values=bytes(values),
            masks=bytes(masks),
            range_start=spec.range_start,
            range_end=spec.range_end,
            alignment=spec.alignment,
            expected_count=spec.expected_count,
        ),
        [],
    )


def compile_patterns(
    specs: list[PatternSpec],
) -> tuple[list[CompiledPattern], list[PatternError]]:
    """Compile every spec; errors are aggregated in spec order."""
    compiled: list[CompiledPattern] = []
    errors: list[PatternError] = []
    for spec in specs:
        pattern, errs = compile_pattern(spec)
        errors.extend(errs)
        if pattern is not None:
            compiled.append(pattern)
    return compiled, errors


def _match_at(data: bytes, offset: int, values: bytes, masks: bytes) -> bool:
    for i, mask in enumerate(masks):
        if (data[offset + i] & mask) != values[i]:
            return False
    return True


def scan_patterns(segments: list[Segment], patterns: list[CompiledPattern]) -> ScanResult:
    """Scan the contiguous ``segments`` for every compiled pattern.

    A match must fit entirely inside one segment (never across a gap);
    its start address must satisfy the pattern's range/alignment
    constraints. Overlapping matches are all reported.
    """
    reports: list[PatternReport] = []
    matches: list[PatternMatch] = []
    for pattern in patterns:
        addresses: list[int] = []
        for segment in segments:
            if pattern.length > len(segment.data):
                continue
            lo = segment.start
            hi = segment.end - pattern.length  # last start keeping the match inside
            if pattern.range_start is not None and pattern.range_end is not None:
                lo = max(lo, pattern.range_start)
                hi = min(hi, pattern.range_end - 1)
            if lo > hi:
                continue
            first = (
                lo
                if lo % pattern.alignment == 0
                else lo + (pattern.alignment - lo % pattern.alignment)
            )
            for address in range(first, hi + 1, pattern.alignment):
                offset = address - segment.start
                if _match_at(segment.data, offset, pattern.values, pattern.masks):
                    addresses.append(address)
                    matches.append(
                        PatternMatch(
                            pattern_index=pattern.index,
                            address=address,
                            size=pattern.length,
                            data=segment.data[offset : offset + pattern.length],
                        )
                    )
        hit_count = len(addresses)
        if pattern.expected_count is not None and hit_count != pattern.expected_count:
            status = COUNT_MISMATCH
        elif hit_count == 0:
            status = ABSENT
        elif hit_count == 1:
            status = UNIQUE
        else:
            status = AMBIGUOUS
        reports.append(
            PatternReport(
                name=pattern.name,
                index=pattern.index,
                source=pattern.source,
                status=status,
                hit_count=hit_count,
                expected_count=pattern.expected_count,
                compiled_length=pattern.length,
                range_start=pattern.range_start,
                range_end=pattern.range_end,
                alignment=pattern.alignment,
                match_addresses=addresses,
            )
        )
    matches.sort(key=lambda m: (m.address, m.pattern_index))
    return ScanResult(
        ok=all(report.status != COUNT_MISMATCH for report in reports),
        reports=reports,
        matches=matches,
    )
