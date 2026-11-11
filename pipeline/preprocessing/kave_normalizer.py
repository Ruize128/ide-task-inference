"""Map KaVE event records and their context to the canonical IDE event schema."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pipeline.preprocessing.canonical_event import (
    CanonicalEvent,
    CanonicalEventType,
    DebugAction,
    NavigationKind,
    VcsAction,
)
from pipeline.preprocessing.kave_parser import classify_command


# Normalizer class

class KaVENormalizer:
    """
    Converts raw KaVE event dicts (output of KaVEParser) into CanonicalEvent
    instances with cross-platform semantics.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Convert one record from KaVEParser into a canonical event."""
        kave_type: str = raw["kave_type"]

        # Dispatch to the appropriate handler based on KaVE event type
        handler = self._get_handler(kave_type)
        return handler(raw)

    def _get_handler(self, kave_type: str):
        """Return the handler method for a given KaVE event type."""
        handlers: dict[str, Any] = {
            "command": self._normalize_command,
            "navigation": self._normalize_navigation,
            "edit": self._normalize_edit,
            "window": self._normalize_window,
            "document": self._normalize_document,
            "solution": self._normalize_solution,
            "ide_state": self._normalize_ide_state,
            "build": self._normalize_build,
            "test": self._normalize_test,
            "debugger": self._normalize_debugger,
            "version_control": self._normalize_version_control,
            "completion": self._normalize_completion,
            "activity": self._normalize_activity,
            "user_profile": self._normalize_user_profile,
        }
        return handlers.get(kave_type, self._normalize_unknown)

    # ------------------------------------------------------------------
    # Common field extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_base(
        raw: dict[str, Any],
        canonical_type: CanonicalEventType,
    ) -> CanonicalEvent:
        """Initialize fields shared by all canonical events."""
        # Use IDE Session UUID as session_id; fall back to a generated one
        session_id = raw.get("session_id") or "unknown"

        # Parse timestamp; use epoch as fallback
        triggered_at = raw.get("triggered_at")
        if not isinstance(triggered_at, datetime):
            triggered_at = datetime(1970, 1, 1, tzinfo=timezone.utc)

        # Parse duration string like "00:00:00.0004995" to milliseconds
        duration_ms = _parse_duration_to_ms(raw.get("duration"))

        return CanonicalEvent(
            event_id=str(uuid.uuid4()),
            canonical_type=canonical_type,
            timestamp=triggered_at,
            session_id=session_id,
            duration_ms=duration_ms,
            active_file=raw.get("active_document"),
            metadata={"_kave_type_full": raw.get("kave_type_full", "")},
        )

    # ------------------------------------------------------------------
    # Per-type normalizers
    # ------------------------------------------------------------------

    def _normalize_command(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Classify the Visual Studio CommandId, falling back to COMMAND_OTHER."""
        payload: dict[str, Any] = raw.get("payload", {})
        command_id: str = payload.get("CommandId", "")

        classification = classify_command(command_id)
        primary = classification["primary"]

        type_map: dict[str, CanonicalEventType] = {
            "build": CanonicalEventType.BUILD,
            "debug": CanonicalEventType.DEBUG,
            "test": CanonicalEventType.TEST_RUN,
            "vcs": CanonicalEventType.VCS,
            "search": CanonicalEventType.SEARCH,
            "navigation": CanonicalEventType.NAVIGATE_STRUCTURAL,
        }
        canonical_type = type_map.get(primary, CanonicalEventType.COMMAND_OTHER)

        event = self._make_base(raw, canonical_type)
        event.command_category = primary
        event.metadata["vs_command_id"] = command_id  # Preserve original for debugging

        if canonical_type == CanonicalEventType.DEBUG:
            event.debug_action = DebugAction.UNKNOWN  # CommandEvents lack detail
        elif canonical_type == CanonicalEventType.VCS:
            event.vcs_action = _infer_vcs_action(command_id)

        return event

    def _normalize_navigation(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Map navigation actions to structural navigation subtypes."""
        payload: dict[str, Any] = raw.get("payload", {})
        nav_type: int = payload.get("TypeOfNavigation", -1)

        # Map KaVE TypeOfNavigation enum values to NavigationKind
        # Values: 0=Unknown, 1=NavigateTo, 2=GoToDefinition,
        #         3=FindUsages, 4=TypeHierarchy, 5=CallHierarchy
        nav_kind_map: dict[int, NavigationKind] = {
            1: NavigationKind.UNKNOWN,  # NavigateTo is too generic
            2: NavigationKind.GO_TO_DEFINITION,
            3: NavigationKind.FIND_USAGES,
            4: NavigationKind.TYPE_HIERARCHY,
            5: NavigationKind.CALL_HIERARCHY,
        }
        nav_kind = nav_kind_map.get(nav_type, NavigationKind.UNKNOWN)

        event = self._make_base(raw, CanonicalEventType.NAVIGATE_STRUCTURAL)
        event.navigation_kind = nav_kind
        event.target_location = payload.get("Location") or payload.get("Target")

        return event

    def _normalize_edit(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Extract member counts and enclosing type from the syntax tree.

        KaVE edit records leave character change counts unset.
        """
        payload: dict[str, Any] = raw.get("payload", {})
        context2: dict[str, Any] = payload.get("Context2", {})

        sst: dict[str, Any] = context2.get("SST", {})

        methods = sst.get("Methods", [])
        fields = sst.get("Fields", [])
        properties = sst.get("Properties", [])
        total_members = len(methods) + len(fields) + len(properties)

        event = self._make_base(raw, CanonicalEventType.EDIT)
        event.change_kind = "MODIFY"
        event.chars_added = None  # Not available in KaVE EditEvents
        event.chars_removed = None

        enclosing_type = sst.get("EnclosingType", "")
        if enclosing_type:
            event.active_symbol = enclosing_type

        event.metadata["sst_method_count"] = len(methods)
        event.metadata["sst_field_count"] = len(fields)
        event.metadata["sst_property_count"] = len(properties)

        return event

    def _normalize_window(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Represent window changes as file switches."""
        event = self._make_base(raw, CanonicalEventType.FILE_SWITCH)
        return event

    def _normalize_document(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Map action 0 to file open, 1 to close, and other values to switch."""
        payload: dict[str, Any] = raw.get("payload", {})
        action: int = payload.get("Action", -1)

        if action == 0:
            ctype = CanonicalEventType.FILE_OPEN
        elif action == 1:
            ctype = CanonicalEventType.FILE_CLOSE
        else:
            ctype = CanonicalEventType.FILE_SWITCH

        event = self._make_base(raw, ctype)
        return event

    def _normalize_solution(self, raw: dict[str, Any]) -> CanonicalEvent:
        """
        Normalize a SolutionEvent (project/solution open/close/add/remove).
        Mapped to COMMAND_OTHER since it's infrastructure-level activity.
        """
        event = self._make_base(raw, CanonicalEventType.COMMAND_OTHER)
        event.command_category = "solution_management"
        return event

    def _normalize_ide_state(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Map lifecycle phase 0 to session start and other phases to session end."""
        payload: dict[str, Any] = raw.get("payload", {})
        phase: int = payload.get("IDELifecyclePhase", -1)

        if phase == 0:
            ctype = CanonicalEventType.SESSION_START
        else:
            ctype = CanonicalEventType.SESSION_END

        event = self._make_base(raw, ctype)
        return event

    def _normalize_build(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Summarize target outcomes. Build success is unknown when there are no targets."""
        payload: dict[str, Any] = raw.get("payload", {})
        targets: list[dict[str, Any]] = payload.get("Targets", [])

        # Determine overall build success (all targets must succeed)
        all_success = all(t.get("Successful", False) for t in targets) if targets else None

        event = self._make_base(raw, CanonicalEventType.BUILD)
        event.build_target = payload.get("Scope", "unknown")
        event.build_success = all_success
        event.metadata["build_target_count"] = len(targets)
        event.metadata["build_action"] = payload.get("Action", "")

        return event

    def _normalize_test(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Extract test results and retain the original payload."""
        payload: dict[str, Any] = raw.get("payload", {})

        event = self._make_base(raw, CanonicalEventType.TEST_RUN)
        # KaVE TestEvent structure varies; preserve raw payload for analysis
        event.test_results = payload.get("Tests", [])
        event.metadata["test_event_raw"] = payload

        return event

    def _normalize_debugger(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Extract the debugger action when available."""
        payload: dict[str, Any] = raw.get("payload", {})

        event = self._make_base(raw, CanonicalEventType.DEBUG)
        event.debug_action = DebugAction.UNKNOWN

        # Attempt to infer debug action from payload
        debug_mode = payload.get("DebugMode", "")
        if "break" in debug_mode.lower():
            event.debug_action = DebugAction.BREAKPOINT_HIT
        elif "step" in debug_mode.lower():
            event.debug_action = DebugAction.STEP_OVER

        event.metadata["debug_event_raw"] = payload
        return event

    def _normalize_version_control(self, raw: dict[str, Any]) -> CanonicalEvent:
        """
        Normalize a VersionControlEvent.

        Contains Actions[] with ActionType enum values for commit, push, etc.
        """
        payload: dict[str, Any] = raw.get("payload", {})
        actions: list[dict[str, Any]] = payload.get("Actions", [])

        event = self._make_base(raw, CanonicalEventType.VCS)
        event.metadata["vcs_action_count"] = len(actions)

        if actions:
            # Extract the first action type
            action_type: int = actions[0].get("ActionType", -1)
            event.vcs_action = _map_vcs_action_type(action_type)

        return event

    def _normalize_completion(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Record code completion and the number of selections."""
        event = self._make_base(raw, CanonicalEventType.CODE_COMPLETION)

        payload: dict[str, Any] = raw.get("payload", {})
        selections = payload.get("Selections", [])
        event.metadata["completion_selection_count"] = len(selections)

        return event

    def _normalize_activity(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Map user-initiated activity to COMMAND_OTHER and other activity to INACTIVITY."""
        triggered_by = raw.get("triggered_by")
        if triggered_by == 1:  # User-initiated activity
            ctype = CanonicalEventType.COMMAND_OTHER  # Generic user action
        else:
            ctype = CanonicalEventType.INACTIVITY

        event = self._make_base(raw, ctype)
        return event

    def _normalize_user_profile(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Keep profile metadata in a COMMAND_OTHER event."""
        event = self._make_base(raw, CanonicalEventType.COMMAND_OTHER)
        event.command_category = "user_profile"
        event.metadata["profile_raw"] = raw.get("payload", {})
        return event

    def _normalize_unknown(self, raw: dict[str, Any]) -> CanonicalEvent:
        """Preserve the original type of an unrecognized event."""
        event = self._make_base(raw, CanonicalEventType.COMMAND_OTHER)
        event.command_category = "unknown"
        event.metadata["unknown_type"] = raw.get("kave_type_full", "")
        return event


# Helper functions (module-level, reusable by other normalizers)

def _parse_duration_to_ms(duration_str: str | None) -> int | None:
    """Parse a .NET TimeSpan, truncating fractions to milliseconds."""
    if not duration_str:
        return None
    try:
        parts = duration_str.split(":")
        if len(parts) != 3:
            return None
        hours = int(parts[0])
        minutes = int(parts[1])
        sec_frac = parts[2]
        if "." in sec_frac:
            seconds_str, frac_str = sec_frac.split(".", 1)
            seconds = int(seconds_str)
            # Fractional seconds to milliseconds (first 3 digits)
            frac_str = frac_str[:3].ljust(3, "0")
            milliseconds = int(frac_str)
        else:
            seconds = int(sec_frac)
            milliseconds = 0
        total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds
        return total_ms
    except (ValueError, IndexError):
        return None


def _infer_vcs_action(command_id: str) -> VcsAction:
    """Infer VcsAction from a VS CommandId string."""
    cmd_lower = command_id.lower()
    if "commit" in cmd_lower:
        return VcsAction.COMMIT
    if "push" in cmd_lower:
        return VcsAction.PUSH
    if "pull" in cmd_lower:
        return VcsAction.PULL
    if "merge" in cmd_lower:
        return VcsAction.MERGE
    if "revert" in cmd_lower:
        return VcsAction.REVERT
    if "branch" in cmd_lower:
        return VcsAction.BRANCH
    if "stash" in cmd_lower:
        return VcsAction.STASH
    return VcsAction.UNKNOWN


def _map_vcs_action_type(action_type: int) -> VcsAction:
    """Map known KaVE VCS action codes and use UNKNOWN for other values."""
    mapping: dict[int, VcsAction] = {
        1: VcsAction.COMMIT,
        2: VcsAction.PUSH,
        3: VcsAction.PULL,
        4: VcsAction.MERGE,
        5: VcsAction.REVERT,
        6: VcsAction.BRANCH,
        7: VcsAction.STASH,
    }
    return mapping.get(action_type, VcsAction.UNKNOWN)

