"""Input loading utilities for processed EnrichedEventStreams artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pipeline.process_mining.config import ProcessMiningConfig


EventDict = dict[str, Any]
SessionEvents = list[EventDict]


def _is_selected_by_fraction(key: object, fraction: float | None, seed: int) -> bool:
    """Return a stable sampling decision for a session-level key."""
    if fraction is None or fraction >= 1.0:
        return True
    if fraction <= 0.0:
        return False
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    sample_value = int.from_bytes(digest, byteorder="big", signed=False) / 2**64
    return sample_value < fraction


def iter_jsonl(path: Path) -> Iterator[Any]:
    """Yield records from JSONL, including session lists and individual events."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_session_event_lists(config: ProcessMiningConfig) -> Iterator[SessionEvents]:
    """Read grouped sessions when available, or group canonical events by session ID."""
    if config.sessions_path.exists():
        yielded = 0
        with config.sessions_path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                # The processed file may be ordered by source/archive. Stride
                # sampling gives quick notebook runs a broader dataset view than
                # simply taking the first N sessions. We skip before json.loads
                # so large skipped session lines do not pay parsing cost.
                if line_index < config.session_offset:
                    continue
                if (line_index - config.session_offset) % max(1, config.session_stride) != 0:
                    continue
                # Fractional sampling happens before json.loads so skipped
                # sessions do not add parsing time or memory pressure. The
                # whole downstream pipeline then uses the same sampled sessions.
                if not _is_selected_by_fraction(line_index, config.session_sample_fraction, config.session_sample_seed):
                    continue

                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, list):
                    continue
                if len(obj) < config.min_events_per_session:
                    continue
                yield obj
                yielded += 1
                if config.max_sessions is not None and yielded >= config.max_sessions:
                    break
        return

    if not config.canonical_events_path.exists():
        raise FileNotFoundError(
            "Neither sessions.jsonl nor canonical_events.jsonl was found. "
            f"Checked: {config.sessions_path} and {config.canonical_events_path}"
        )

    # Fallback path for portability. It is intentionally simple and best used
    # for samples, because grouping a full multi-GB canonical-events file may
    # require substantial memory if events are not session-contiguous.
    grouped: dict[str, SessionEvents] = defaultdict(list)
    for event in iter_jsonl(config.canonical_events_path):
        session_id = str(event.get("session_id", "unknown"))
        if not _is_selected_by_fraction(session_id, config.session_sample_fraction, config.session_sample_seed):
            continue
        grouped[session_id].append(event)
        if config.max_sessions is not None and len(grouped) >= config.max_sessions:
            break

    for events in grouped.values():
        if len(events) >= config.min_events_per_session:
            yield events


def load_session_sample(config: ProcessMiningConfig, limit: int = 5) -> list[SessionEvents]:
    """Load a small sample for notebook inspection."""
    sample: list[SessionEvents] = []
    for events in iter_session_event_lists(config):
        sample.append(events)
        if len(sample) >= limit:
            break
    return sample