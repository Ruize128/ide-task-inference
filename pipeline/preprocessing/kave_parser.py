"""Stream KaVE JSON events from ZIP archives.

Use the .NET type in `$type` to classify events and retain type-specific fields in payload.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any


# Constants - known KaVE event type markers (extracted from $type field)

# The $type field contains fully-qualified .NET type names.
# We match against known suffixes to classify events.
_KAVE_TYPE_MARKERS: dict[str, str] = {
    "CommandEvent": "command",
    "NavigationEvent": "navigation",
    "CompletionEvent": "completion",
    "EditEvent": "edit",
    "WindowEvent": "window",
    "ActivityEvent": "activity",
    "DocumentEvent": "document",
    "SolutionEvent": "solution",
    "IDEStateEvent": "ide_state",
    "DebuggerEvent": "debugger",
    "BuildEvent": "build",
    "TestEvent": "test",
    "VersionControlEvent": "version_control",
    "UserProfileEvent": "user_profile",
    "SystemEvent": "system",
}

# Known Visual Studio command IDs used to categorize CommandEvents.
_BUILD_COMMANDS: set[str] = {
    "BuildSolution", "BuildProject", "BuildOnlyProject",
    "RebuildSolution", "RebuildProject", "CleanSolution",
    "BuildSelection", "Compile",
}
_DEBUG_COMMANDS: set[str] = {
    "DebugStart", "DebugStartWithoutDebugging",
    "DebugStepOver", "DebugStepInto", "DebugStepOut",
    "DebugContinue", "DebugStop", "DebugRestart",
    "DebugToggleBreakpoint", "DebugAttachToProcess",
}
_TEST_COMMANDS: set[str] = {
    "TestExplorerRunAllTests", "TestRun", "TestDebug",
    "RunTestsInCurrentContext", "RunAllTestsInSolution",
    "TestExplorerRunTest",
}
_VCS_COMMANDS: set[str] = {
    "Commit", "TeamGitCommit", "TeamGitPush", "TeamGitPull",
    "TeamGitMerge", "TeamGitRevert", "TeamGitFetch",
    "GitCommands", "FileGitHistory",
}
_SEARCH_COMMANDS: set[str] = {
    "EditFind", "EditFindInFiles", "EditFindNext",
    "EditFindPrevious", "EditFindAllReferences",
    "EditNavigateTo", "EditGoTo", "EditGoToDefinition",
}
_NAVIGATION_COMMANDS: set[str] = {
    "NavigateTo", "EditGoToDefinition", "EditGoToDeclaration",
    "EditGoToImplementation", "EditFindAllReferences",
    "ViewCallHierarchy", "ViewTypeHierarchy",
}


# Parser class

class KaVEParser:
    """Read participant ZIP archives from a KaVE release directory."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_zip_files(self) -> list[Path]:
        """Find ZIP archives recursively in sorted path order."""
        zip_paths = sorted(self.dataset_root.rglob("*.zip"))
        if not zip_paths:
            raise FileNotFoundError(
                f"No .zip files found under {self.dataset_root}"
            )
        return zip_paths

    def iter_events(
        self,
        zip_paths: list[Path] | None = None,
        max_files: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield events from the selected ZIPs, or discover them when omitted.

        max_files limits ZIP archives, not events. Each record contains the event type,
        session ID, timestamp, IDE context, and type-specific payload.
        """
        if zip_paths is None:
            zip_paths = self.discover_zip_files()

        if max_files is not None:
            zip_paths = zip_paths[:max_files]

        for zip_path in zip_paths:
            yield from self._iter_events_from_zip(zip_path)

    def iter_events_from_single_zip(self, zip_path: str | Path) -> Iterator[dict[str, Any]]:
        """
        Iterate over all raw events from a single ZIP file.

        Convenience wrapper for quick exploration.
        """
        yield from self._iter_events_from_zip(Path(zip_path))

    # ------------------------------------------------------------------
    # Internal: ZIP-level parsing
    # ------------------------------------------------------------------

    def _iter_events_from_zip(self, zip_path: Path) -> Iterator[dict[str, Any]]:
        """
        Open a ZIP archive and yield each JSON event inside.

        Each entry in the ZIP is a numbered .json file (0.json, 1.json, ...).
        """
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Sort entries numerically by filename stem
                entries = sorted(
                    zf.infolist(),
                    key=lambda e: int(Path(e.filename).stem)
                    if Path(e.filename).stem.isdigit()
                    else 0,
                )
                for entry in entries:
                    if not entry.filename.endswith(".json"):
                        continue
                    try:
                        raw_json = zf.read(entry).decode("utf-8")
                        event_dict = json.loads(raw_json)
                        normalized = self._normalize_raw_event(event_dict)
                        yield normalized
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        # Report malformed JSON and continue.
                        print(f"[WARN] Skipping corrupted entry {entry.filename} "
                              f"in {zip_path.name}: {exc}")
                        continue
        except zipfile.BadZipFile as exc:
            print(f"[WARN] Skipping bad ZIP: {zip_path.name}: {exc}")

    # ------------------------------------------------------------------
    # Internal: event normalization
    # ------------------------------------------------------------------

    def _normalize_raw_event(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Extract common fields and keep type-specific fields in payload."""

        type_full = raw.get("$type", "")
        kave_type = self._classify_type(type_full)

        session_id = raw.get("IDESessionUUID", "unknown")
        triggered_at = self._parse_timestamp(raw.get("TriggeredAt"))
        triggered_by = raw.get("TriggeredBy")
        duration = raw.get("Duration")
        active_window = raw.get("ActiveWindow")
        active_document = raw.get("ActiveDocument")

        common_keys = {
            "$type", "IDESessionUUID", "TriggeredAt", "TriggeredBy",
            "Duration", "ActiveWindow", "ActiveDocument",
            "KaVEVersion",
        }
        payload = {k: v for k, v in raw.items() if k not in common_keys}

        return {
            "kave_type": kave_type,
            "kave_type_full": type_full,
            "session_id": session_id,
            "triggered_at": triggered_at,
            "triggered_by": triggered_by,
            "duration": duration,
            "active_window": active_window,
            "active_document": active_document,
            "payload": payload,
        }

    @staticmethod
    def _classify_type(type_full: str) -> str:
        """Map the KaVE type to an event category, defaulting to unknown."""
        for marker, label in _KAVE_TYPE_MARKERS.items():
            if marker in type_full:
                return label
        return "unknown"

    @staticmethod
    def _parse_timestamp(raw_ts: str | None) -> datetime | None:
        """Parse KaVE ISO timestamps, trimming excess fractional precision if needed."""
        if not raw_ts:
            return None
        try:
            # Try ISO format with timezone
            return datetime.fromisoformat(raw_ts)
        except ValueError:
            pass
        try:
            # Fallback: truncate to 6 decimal digits for microsecond precision
            # Some KaVE timestamps have 7+ fractional digits
            if "." in raw_ts:
                main, frac_tz = raw_ts.split(".", 1)
                frac_digits = "".join(c for c in frac_tz if c.isdigit())[:6]
                tz_part = frac_tz[len(frac_digits):] if len(frac_tz) > len("".join(c for c in frac_tz if c.isdigit())) else ""
                clean = f"{main}.{frac_digits}{tz_part}"
            else:
                clean = raw_ts
            return datetime.fromisoformat(clean)
        except ValueError:
            return None


# Module-level helpers for command classification

def classify_command(command_id: str) -> dict[str, str]:
    """Return the primary and secondary categories of a Visual Studio command."""
    if command_id in _BUILD_COMMANDS:
        return {"primary": "build", "secondary": "build"}
    if command_id in _DEBUG_COMMANDS:
        return {"primary": "debug", "secondary": "debug"}
    if command_id in _TEST_COMMANDS:
        return {"primary": "test", "secondary": "test"}
    if command_id in _VCS_COMMANDS:
        return {"primary": "vcs", "secondary": "vcs"}
    if command_id in _SEARCH_COMMANDS:
        return {"primary": "search", "secondary": "search"}
    if command_id in _NAVIGATION_COMMANDS:
        return {"primary": "navigation", "secondary": "navigation"}

    # Second-pass heuristics for common patterns
    cmd_lower = command_id.lower()
    if "build" in cmd_lower:
        return {"primary": "build", "secondary": "build"}
    if "debug" in cmd_lower:
        return {"primary": "debug", "secondary": "debug"}
    if "test" in cmd_lower:
        return {"primary": "test", "secondary": "test"}
    if any(vcs in cmd_lower for vcs in ("commit", "push", "pull", "merge", "git")):
        return {"primary": "vcs", "secondary": "vcs"}
    if any(s in cmd_lower for s in ("find", "search", "goto", "navigate")):
        return {"primary": "search", "secondary": "navigation"}

    return {"primary": "other", "secondary": "other"}
