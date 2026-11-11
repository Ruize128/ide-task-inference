"""Canonical event types and fields shared by IDE event processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CanonicalEventType(str, Enum):
    """Canonical IDE event categories. COMMAND_OTHER covers unmapped commands."""

    # --- IDE lifecycle ---
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # --- File operations ---
    FILE_OPEN = "file_open"
    FILE_CLOSE = "file_close"
    FILE_SWITCH = "file_switch"

    # --- Editing ---
    EDIT = "edit"

    # --- Navigation ---
    NAVIGATE_STRUCTURAL = "nav_structural"  # ctrl+click, go-to-definition, etc.
    NAVIGATE_CARET = "nav_caret"  # cursor movement, scrolling

    # --- Tool usage ---
    SEARCH = "search"
    CODE_COMPLETION = "code_completion"

    # --- Specialized tools ---
    BUILD = "build"
    TEST_RUN = "test_run"
    DEBUG = "debug"
    VCS = "vcs"

    # --- Fallback ---
    COMMAND_OTHER = "command_other"  # unrecognized IDE commands
    INACTIVITY = "inactivity"  # user idle
    SYSTEM_EVENT = "system_event"  # OS-level: lock, sleep, screen saver


# Navigation sub-kinds (for NAVIGATE_STRUCTURAL context enrichment)

class NavigationKind(str, Enum):
    """Structured navigation sub-types."""
    GO_TO_DEFINITION = "go_to_definition"
    GO_TO_IMPLEMENTATION = "go_to_implementation"
    GO_TO_TYPE_DECLARATION = "go_to_type_declaration"
    FIND_USAGES = "find_usages"
    CALL_HIERARCHY = "call_hierarchy"
    TYPE_HIERARCHY = "type_hierarchy"
    UNKNOWN = "unknown"


# VCS action sub-kinds

class VcsAction(str, Enum):
    """Version control action sub-types."""
    COMMIT = "commit"
    PUSH = "push"
    PULL = "pull"
    MERGE = "merge"
    REVERT = "revert"
    BRANCH = "branch"
    STASH = "stash"
    UNKNOWN = "unknown"


# Debug action sub-kinds

class DebugAction(str, Enum):
    """Debugger action sub-types."""
    BREAKPOINT_HIT = "breakpoint_hit"
    BREAKPOINT_SET = "breakpoint_set"
    BREAKPOINT_REMOVE = "breakpoint_remove"
    STEP_OVER = "step_over"
    STEP_INTO = "step_into"
    STEP_OUT = "step_out"
    CONTINUE = "continue"
    INSPECT_VARIABLE = "inspect_variable"
    EVALUATE_EXPRESSION = "evaluate_expression"
    UNKNOWN = "unknown"


# CanonicalEvent dataclass - the central data structure

@dataclass(slots=True)
class CanonicalEvent:
    """A normalized IDE event with optional source-specific context.

    The event ID, type, timestamp, and session ID are required.
    """

    # --- Required fields ---
    event_id: str
    canonical_type: CanonicalEventType
    timestamp: datetime
    session_id: str

    # --- Timing ---
    duration_ms: int | None = None
    """Duration of the activity in milliseconds (e.g., build time)."""

    # --- File / symbol context ---
    active_file: str | None = None
    """Path of the currently active source file."""

    active_symbol: str | None = None
    """Fully-qualified name of the symbol under cursor."""

    target_location: str | None = None
    """
    Target of a navigation action, encoded as a URI-like string.
    Format depends on the source IDE but should include file + symbol info.
    """

    # --- Edit-specific context ---
    change_kind: str | None = None
    """Edit change kind: 'INSERT', 'DELETE', 'MODIFY', or None."""

    chars_added: int | None = None
    """Number of characters added in this edit."""

    chars_removed: int | None = None
    """Number of characters removed in this edit."""

    # --- Build-specific context ---
    build_target: str | None = None
    """Build scope: 'project', 'solution', 'file'."""

    build_success: bool | None = None
    """Whether the build completed without errors."""

    # --- Test-specific context ---
    test_results: list[dict[str, Any]] | None = None
    """
    List of individual test results.
    Each dict: {'testName': str, 'result': 'passed'|'failed'|'skipped', 'durationMs': int}
    """

    # --- Navigation-specific context ---
    navigation_kind: NavigationKind | None = None
    """Sub-type of structural navigation."""

    # --- Search-specific context ---
    search_query: str | None = None
    """The search query string."""

    # --- VCS-specific context ---
    vcs_action: VcsAction | None = None
    """Sub-type of version control action."""

    # --- Debug-specific context ---
    debug_action: DebugAction | None = None
    """Sub-type of debugger action."""

    # --- Fallback ---
    command_category: str | None = None
    """
    Broad category for COMMAND_OTHER events.
    Examples: 'refactoring', 'navigation', 'window_management', 'unknown'.
    """

    # --- Extra metadata ---
    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary extra data preserved from the source event for debugging."""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_edit(self) -> bool:
        """Check if this event is an edit action."""
        return self.canonical_type == CanonicalEventType.EDIT

    @property
    def is_navigation(self) -> bool:
        """Check if this event is any kind of navigation."""
        return self.canonical_type in (
            CanonicalEventType.NAVIGATE_STRUCTURAL,
            CanonicalEventType.NAVIGATE_CARET,
        )

    @property
    def is_lifecycle(self) -> bool:
        """Check if this event marks a session boundary."""
        return self.canonical_type in (
            CanonicalEventType.SESSION_START,
            CanonicalEventType.SESSION_END,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a JSON-compatible dict.

        Enums are converted to their string values for portability.
        """
        result: dict[str, Any] = {
            "event_id": self.event_id,
            "canonical_type": self.canonical_type.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
        }
        # Only include non-None optional fields
        optional_fields = [
            "duration_ms",
            "active_file", "active_symbol", "target_location",
            "change_kind", "chars_added", "chars_removed",
            "build_target", "build_success",
            "test_results",
            "navigation_kind", "search_query",
            "vcs_action", "debug_action",
            "command_category",
        ]
        for field_name in optional_fields:
            value = getattr(self, field_name, None)
            if value is not None:
                if isinstance(value, Enum):
                    result[field_name] = value.value
                else:
                    result[field_name] = value
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalEvent":
        """
        Deserialize from a dict (e.g., loaded from JSONL).

        String enum values are converted back to their Enum types.
        """
        # Convert canonical_type string back to enum
        ctype = CanonicalEventType(data["canonical_type"])

        # Convert timestamp string back to datetime
        ts = datetime.fromisoformat(data["timestamp"])

        # Convert sub-enum strings back to enums if present
        nav_kind = None
        if "navigation_kind" in data and data["navigation_kind"] is not None:
            nav_kind = NavigationKind(data["navigation_kind"])

        vcs = None
        if "vcs_action" in data and data["vcs_action"] is not None:
            vcs = VcsAction(data["vcs_action"])

        debug = None
        if "debug_action" in data and data["debug_action"] is not None:
            debug = DebugAction(data["debug_action"])

        return cls(
            event_id=data["event_id"],
            canonical_type=ctype,
            timestamp=ts,
            session_id=data["session_id"],
            duration_ms=data.get("duration_ms"),
            active_file=data.get("active_file"),
            active_symbol=data.get("active_symbol"),
            target_location=data.get("target_location"),
            change_kind=data.get("change_kind"),
            chars_added=data.get("chars_added"),
            chars_removed=data.get("chars_removed"),
            build_target=data.get("build_target"),
            build_success=data.get("build_success"),
            test_results=data.get("test_results"),
            navigation_kind=nav_kind,
            search_query=data.get("search_query"),
            vcs_action=vcs,
            debug_action=debug,
            command_category=data.get("command_category"),
            metadata=data.get("metadata", {}),
        )
