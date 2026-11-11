"""Convert preprocessed event dictionaries into tables for process mining."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Any

import pandas as pd


EVENT_TO_ACTION: dict[str, str] = {
    "session_start": "lifecycle",
    "session_end": "lifecycle",
    "file_open": "file_context",
    "file_close": "file_context",
    "file_switch": "file_context",
    "edit": "editing",
    "nav_structural": "navigation",
    "nav_caret": "navigation",
    "search": "search",
    "code_completion": "completion",
    "build": "build",
    "test_run": "test",
    "debug": "debug",
    "vcs": "vcs",
    "command_other": "command_other",
    "inactivity": "inactivity",
    "system_event": "system",
}


def parse_timestamp(value: Any) -> datetime | None:
    """Parse ISO timestamps and normalize them to timezone-aware UTC."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def file_extension(path: str | None) -> str:
    """Return a lowercase file extension when the active file is parseable."""
    if not path:
        return ""
    # KaVE active documents may contain prefixes such as ``0Doc:CSharp``.
    cleaned = path.split()[-1] if " " in path else path
    return PurePath(cleaned).suffix.lower()


def command_name(event: dict[str, Any]) -> str:
    """Extract a stable command label from metadata if one is present."""
    metadata = event.get("metadata") or {}
    for key in ("vs_command_id", "command_id", "command_name", "button"):
        value = metadata.get(key) or event.get(key)
        if value:
            return str(value)
    return ""


def infer_action_type(canonical_type: str, category: str, command: str) -> str:
    """Refine generic command events using keywords in the original command name."""
    base_action = EVENT_TO_ACTION.get(canonical_type, "unknown")
    if canonical_type != "command_other":
        return base_action

    text = f"{category} {command}".lower()

    if any(token in text for token in (
        "deletebackwards", "backspace", "breakline", "paste", "cut",
        "undo", "redo", "insert", "tab", "edit.delete", "textcontrol.enter",
    )):
        return "editing"

    if any(token in text for token in (
        "charleft", "charright", "lineup", "linedown", "lineend",
        "linestart", "pageup", "pagedown", "wordprevious", "wordnext",
        "selectioncancel", "nextdocument", "previousdocument",
    )):
        return "navigation"

    if any(token in text for token in ("save", "close", "open", "window.")):
        return "file_context"

    if any(token in text for token in ("completion", "postfixtemplates", "completeword")):
        return "completion"

    if any(token in text for token in ("find", "query", "search", "navigate")):
        return "search"

    if any(token in text for token in ("debug", "continue", "stepover", "stepinto", "stepout")):
        return "debug"

    if any(token in text for token in ("refactor", "rename", "extract", "format", "altenter")):
        return "editing"

    return base_action


def canonicalize_session(events: list[dict[str, Any]], session_index: int) -> pd.DataFrame:
    """Normalize one session into a table and keep source event IDs for traceability."""
    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        timestamp = parse_timestamp(event.get("timestamp"))
        if timestamp is None:
            continue

        canonical_type = str(event.get("canonical_type", "unknown"))
        active_file = event.get("active_file")
        metadata = event.get("metadata") or {}
        category = event.get("command_category") or ""
        command = command_name(event)
        action_type = infer_action_type(canonical_type, category, command)

        rows.append(
            {
                "event_id": str(event.get("event_id", f"s{session_index}_e{event_index}")),
                "session_id": str(event.get("session_id", f"session_{session_index}")),
                "session_index": session_index,
                "event_index": event_index,
                "timestamp": timestamp,
                "canonical_type": canonical_type,
                "action_type": action_type,
                "active_file": active_file or "",
                "file_extension": file_extension(active_file),
                "active_symbol": event.get("active_symbol") or "",
                "command_category": category,
                "command_name": command,
                "change_kind": event.get("change_kind") or "",
                "chars_added": event.get("chars_added"),
                "chars_removed": event.get("chars_removed"),
                "build_success": event.get("build_success"),
                "build_target": event.get("build_target") or "",
                "test_results": event.get("test_results") or [],
                "navigation_kind": event.get("navigation_kind") or "",
                "target_location": event.get("target_location") or "",
                "search_query": event.get("search_query") or "",
                "active_window": event.get("active_window") or metadata.get("active_window", ""),
                "vcs_action": event.get("vcs_action") or "",
                "debug_action": event.get("debug_action") or "",
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["timestamp", "event_index"]).reset_index(drop=True)
    return df


def canonical_event_profile(events: pd.DataFrame) -> dict[str, Any]:
    """Return compact profile statistics for a canonical event dataframe."""
    if events.empty:
        return {"events": 0, "sessions": 0, "top_event_types": []}
    counts = Counter(events["canonical_type"].tolist())
    return {
        "events": int(len(events)),
        "sessions": int(events["session_id"].nunique()),
        "top_event_types": counts.most_common(15),
    }
