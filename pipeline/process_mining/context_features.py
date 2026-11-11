
"""Build context features for activity segmentation and segment clustering."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log1p
from pathlib import PurePath
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from pipeline.process_mining.case_segmentation import activity_family


SOURCE_EXTENSIONS = {
    ".cs", ".java", ".kt", ".py", ".js", ".ts", ".tsx", ".jsx", ".cpp",
    ".c", ".h", ".hpp", ".go", ".rs", ".rb", ".php", ".swift", ".scala",
}
TEST_HINTS = ("test", "tests", "spec", "fixture")
CONFIG_EXTENSIONS = {
    ".json", ".xml", ".yml", ".yaml", ".config", ".props", ".targets",
    ".csproj", ".sln", ".gradle", ".toml", ".ini",
}
DOC_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".htm"}
GENERATED_HINTS = ("generated", "designer", ".g.", ".designer.", "obj/", "bin/")


@dataclass(slots=True)
class FeatureReductionResult:
    """Container for sparse features and their reduced dense representation."""

    feature_matrix: Any
    reduced: np.ndarray
    feature_names: list[str]
    vectorizer: Any
    selector: Any
    reducer: Any | None
    scaler: Any | None
    n_components: int
    # Sparse-safe per-column scaling applied BEFORE TruncatedSVD so large-
    # magnitude numeric columns cannot dominate the leading singular vectors.
    pre_scaler: Any | None = None


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "nan", "null"}:
        return ""
    return text


def file_extension(path: Any) -> str:
    text = safe_text(path)
    if not text:
        return ""
    cleaned = text.split()[-1] if " " in text else text
    return PurePath(cleaned).suffix.lower()


def file_role(path: Any, extension: str | None = None) -> str:
    text = safe_text(path).replace("\\", "/").lower()
    ext = extension or file_extension(text)
    if not text:
        return "unknown"
    if any(hint in text for hint in GENERATED_HINTS):
        return "generated"
    if any(hint in text for hint in TEST_HINTS):
        return "test"
    if ext in CONFIG_EXTENSIONS:
        return "config"
    if ext in DOC_EXTENSIONS:
        return "doc"
    if ext in SOURCE_EXTENSIONS:
        return "source"
    return "other"


def command_family(command_name: Any, command_category: Any = "", action_type: Any = "") -> str:
    text = f"{safe_text(command_category)} {safe_text(command_name)} {safe_text(action_type)}".lower()
    if not text.strip():
        return "unknown"
    if any(token in text for token in ("debug", "stepover", "stepinto", "continue", "breakpoint")):
        return "debug"
    if any(token in text for token in ("build", "compile", "rebuild", "clean")):
        return "build"
    if any(token in text for token in ("test", "nunit", "mstest", "xunit")):
        return "test"
    if any(token in text for token in ("git", "commit", "push", "pull", "merge", "branch", "vcs")):
        return "vcs"
    if any(token in text for token in ("refactor", "rename", "extract", "format", "altenter")):
        return "refactor"
    if any(token in text for token in ("completion", "completeword", "postfixtemplates")):
        return "completion"
    if any(token in text for token in ("find", "search", "query", "navigate")):
        return "search_navigation"
    if any(token in text for token in ("open", "close", "save", "window", "document")):
        return "file_window"
    if any(token in text for token in ("delete", "backspace", "paste", "cut", "undo", "redo", "insert", "enter")):
        return "editing"
    if any(token in text for token in ("project", "solution", "package", "nuget", "reference")):
        return "project_context"
    return "other"


def symbol_kind(symbol: Any) -> str:
    text = safe_text(symbol)
    if not text:
        return "unknown"
    lowered = text.lower()
    if "method" in lowered or "()" in lowered:
        return "method"
    if "property" in lowered:
        return "property"
    if "field" in lowered:
        return "field"
    if "class" in lowered or "." in text:
        return "type_or_member"
    return "other"


def enrich_canonical_context(events: pd.DataFrame) -> pd.DataFrame:
    """Normalize context fields used in segmentation and clustering."""
    if events.empty:
        return events.copy()

    result = events.copy()
    for column in [
        "active_file", "active_symbol", "command_name", "command_category",
        "action_type", "canonical_type", "debug_action", "vcs_action",
        "navigation_kind", "target_location", "build_target", "search_query",
        "active_window",
    ]:
        if column not in result.columns:
            result[column] = ""

    result["file_extension"] = result["active_file"].map(file_extension)
    result["file_role"] = [file_role(path, ext) for path, ext in zip(result["active_file"], result["file_extension"], strict=False)]
    result["command_family"] = [
        command_family(command, category, action)
        for command, category, action in zip(
            result["command_name"], result["command_category"], result["action_type"], strict=False
        )
    ]
    result["symbol_kind"] = result["active_symbol"].map(symbol_kind)
    result["has_active_file"] = result["active_file"].map(lambda value: 1.0 if safe_text(value) else 0.0)
    result["has_symbol"] = result["active_symbol"].map(lambda value: 1.0 if safe_text(value) else 0.0)
    result["has_target_location"] = result["target_location"].map(lambda value: 1.0 if safe_text(value) else 0.0)
    return result


def counter_items(value: Any) -> list[tuple[str, float]]:
    if not isinstance(value, dict):
        return []
    items: list[tuple[str, float]] = []
    for key, count in value.items():
        name = safe_text(key)
        if not name:
            continue
        try:
            numeric = float(count)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric) or numeric <= 0:
            continue
        items.append((name, numeric))
    return items


def normalized_counts(values: list[str], prefix: str) -> dict[str, float]:
    clean = [safe_text(value) for value in values if safe_text(value)]
    total = max(1, len(clean))
    counts = Counter(clean)
    return {f"{prefix}={name}": count / total for name, count in counts.items()}


def entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = np.array([count / total for count in counts.values() if count > 0], dtype=float)
    return float(-(probs * np.log2(probs)).sum())


def add_count_features(features: dict[str, float], counts: Any, prefix: str, normalize: bool = True) -> None:
    items = counter_items(counts)
    total = sum(value for _, value in items) if normalize else 1.0
    total = max(1.0, total)
    for name, value in items:
        features[f"{prefix}={name}"] = float(value / total if normalize else value)


def activity_feature_dicts(activities: pd.DataFrame) -> list[dict[str, float]]:
    """Build broad context-aware feature dictionaries for activity bursts."""
    if activities.empty:
        return []

    rows = activities.sort_values(["session_index", "start_time", "activity_index"]).to_dict("records")
    features: list[dict[str, float]] = []
    previous_by_session: dict[int, dict[str, Any]] = {}

    for row in rows:
        session_index = int(row.get("session_index", -1))
        activity = safe_text(row.get("activity")) or "UNKNOWN_ACTIVITY"
        family = activity_family(activity)
        previous = previous_by_session.get(session_index)
        previous_activity = safe_text(previous.get("activity")) if previous else "SESSION_START"
        previous_file = safe_text(previous.get("dominant_file")) if previous else ""
        current_file = safe_text(row.get("dominant_file"))
        gap_seconds = 0.0
        if previous:
            gap_seconds = max(0.0, (pd.Timestamp(row["start_time"]) - pd.Timestamp(previous["end_time"])).total_seconds())

        item: dict[str, float] = {
            f"activity={activity}": 1.0,
            f"family={family}": 1.0,
            f"prev_activity={previous_activity}": 1.0,
            f"transition={previous_activity}->{activity}": 1.0,
            "num:log_duration_seconds": log1p(float(row.get("duration_seconds") or 0.0)),
            "num:log_event_count": log1p(float(row.get("event_count") or 0.0)),
            "num:log_gap_seconds": log1p(gap_seconds),
            "num:file_count": float(row.get("file_count") or 0.0),
            "num:file_shift_from_previous": 1.0 if previous_file and current_file and previous_file != current_file else 0.0,
        }
        add_count_features(item, row.get("source_event_types"), "source_event")
        add_count_features(item, row.get("source_action_types"), "source_action")
        add_count_features(item, row.get("file_extension_counts"), "file_ext")
        add_count_features(item, row.get("file_role_counts"), "file_role")
        add_count_features(item, row.get("command_family_counts"), "command_family")
        add_count_features(item, row.get("command_name_counts"), "command_name")
        add_count_features(item, row.get("command_category_counts"), "command_category")
        add_count_features(item, row.get("symbol_kind_counts"), "symbol_kind")
        add_count_features(item, row.get("debug_action_counts"), "debug_action")
        add_count_features(item, row.get("vcs_action_counts"), "vcs_action")
        add_count_features(item, row.get("navigation_kind_counts"), "navigation_kind")
        add_count_features(item, row.get("change_kind_counts"), "change_kind")
        add_count_features(item, row.get("build_success_counts"), "build_success")
        features.append(item)
        previous_by_session[session_index] = row

    return features



def iter_activity_feature_dict_batches(
    activities: pd.DataFrame,
    batch_size: int = 100_000,
    sort: bool = True,
) -> Any:
    """Yield feature batches to limit memory use on large activity tables."""
    if activities.empty:
        return

    frame = activities
    if sort:
        frame = activities.sort_values(["session_index", "start_time", "activity_index"])

    previous_by_session: dict[int, dict[str, Any]] = {}
    batch: list[dict[str, float]] = []
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        session_index = int(row_dict.get("session_index", -1))
        activity = safe_text(row_dict.get("activity")) or "UNKNOWN_ACTIVITY"
        family = activity_family(activity)
        previous = previous_by_session.get(session_index)
        previous_activity = safe_text(previous.get("activity")) if previous else "SESSION_START"
        previous_file = safe_text(previous.get("dominant_file")) if previous else ""
        current_file = safe_text(row_dict.get("dominant_file"))
        gap_seconds = 0.0
        if previous:
            gap_seconds = max(
                0.0,
                (pd.Timestamp(row_dict["start_time"]) - pd.Timestamp(previous["end_time"])).total_seconds(),
            )

        item: dict[str, float] = {
            f"activity={activity}": 1.0,
            f"family={family}": 1.0,
            f"prev_activity={previous_activity}": 1.0,
            f"transition={previous_activity}->{activity}": 1.0,
            "num:log_duration_seconds": log1p(float(row_dict.get("duration_seconds") or 0.0)),
            "num:log_event_count": log1p(float(row_dict.get("event_count") or 0.0)),
            "num:log_gap_seconds": log1p(gap_seconds),
            "num:file_count": float(row_dict.get("file_count") or 0.0),
            "num:file_shift_from_previous": 1.0 if previous_file and current_file and previous_file != current_file else 0.0,
        }
        add_count_features(item, row_dict.get("source_event_types"), "source_event")
        add_count_features(item, row_dict.get("source_action_types"), "source_action")
        add_count_features(item, row_dict.get("file_extension_counts"), "file_ext")
        add_count_features(item, row_dict.get("file_role_counts"), "file_role")
        add_count_features(item, row_dict.get("command_family_counts"), "command_family")
        add_count_features(item, row_dict.get("command_name_counts"), "command_name")
        add_count_features(item, row_dict.get("command_category_counts"), "command_category")
        add_count_features(item, row_dict.get("symbol_kind_counts"), "symbol_kind")
        add_count_features(item, row_dict.get("debug_action_counts"), "debug_action")
        add_count_features(item, row_dict.get("vcs_action_counts"), "vcs_action")
        add_count_features(item, row_dict.get("navigation_kind_counts"), "navigation_kind")
        add_count_features(item, row_dict.get("change_kind_counts"), "change_kind")
        add_count_features(item, row_dict.get("build_success_counts"), "build_success")

        batch.append(item)
        previous_by_session[session_index] = row_dict
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def fit_reduce_feature_dicts(
    feature_dicts: list[dict[str, float]],
    max_components: int = 32,
    random_state: int = 42,
) -> FeatureReductionResult:
    """Vectorize sparse features, filter by variance, and fit a TruncatedSVD representation."""
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler
    from sklearn.decomposition import TruncatedSVD

    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(feature_dicts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    selector = VarianceThreshold(threshold=0.0)
    selected = selector.fit_transform(matrix)
    selected_names = [name for name, keep in zip(feature_names, selector.get_support(), strict=False) if keep]

    if selected.shape[0] == 0 or selected.shape[1] == 0:
        return FeatureReductionResult(matrix, np.empty((selected.shape[0], 0)), selected_names, vectorizer, selector, None, None, 0)

    # MaxAbsScaler keeps sparsity while putting every column on a comparable
    # [-1, 1] scale, so SVD components reflect co-occurrence structure rather
    # than whichever raw column happens to have the largest magnitude.
    pre_scaler = MaxAbsScaler()
    scaled = pre_scaler.fit_transform(selected)

    n_components = min(max_components, scaled.shape[0] - 1, scaled.shape[1] - 1)
    if n_components >= 2:
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(scaled)
    else:
        reducer = None
        reduced = scaled.toarray() if sparse.issparse(scaled) else np.asarray(scaled)
        n_components = reduced.shape[1]

    scaler = StandardScaler()
    reduced = scaler.fit_transform(reduced) if reduced.size else reduced
    return FeatureReductionResult(selected, reduced, selected_names, vectorizer, selector, reducer, scaler, int(n_components), pre_scaler)



def transform_feature_dicts(
    feature_dicts: list[dict[str, float]],
    reduction: FeatureReductionResult,
) -> np.ndarray:
    """Transform feature dictionaries with an already fitted reduction stack."""
    if not feature_dicts:
        return np.empty((0, reduction.n_components), dtype=np.float32)

    matrix = reduction.vectorizer.transform(feature_dicts)
    selected = reduction.selector.transform(matrix)
    if reduction.pre_scaler is not None:
        selected = reduction.pre_scaler.transform(selected)

    if reduction.reducer is not None:
        reduced = reduction.reducer.transform(selected)
    else:
        reduced = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)

    if reduction.scaler is not None and reduced.size:
        reduced = reduction.scaler.transform(reduced)
    return np.asarray(reduced, dtype=np.float32)


def aggregate_segment_context(segment: dict[str, Any], activity_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate activity-level context dictionaries into one segment summary."""
    activity_rows = [activity_lookup[activity_id] for activity_id in segment.get("activity_ids", []) if activity_id in activity_lookup]
    file_extensions: Counter[str] = Counter()
    file_roles: Counter[str] = Counter()
    command_families: Counter[str] = Counter()
    command_names: Counter[str] = Counter()
    command_categories: Counter[str] = Counter()
    symbol_kinds: Counter[str] = Counter()
    debug_actions: Counter[str] = Counter()
    vcs_actions: Counter[str] = Counter()
    navigation_kinds: Counter[str] = Counter()
    source_events: Counter[str] = Counter()
    source_actions: Counter[str] = Counter()
    files: list[str] = []
    symbols: list[str] = []

    for activity in activity_rows:
        for counter, key in [
            (file_extensions, "file_extension_counts"),
            (file_roles, "file_role_counts"),
            (command_families, "command_family_counts"),
            (command_names, "command_name_counts"),
            (command_categories, "command_category_counts"),
            (symbol_kinds, "symbol_kind_counts"),
            (debug_actions, "debug_action_counts"),
            (vcs_actions, "vcs_action_counts"),
            (navigation_kinds, "navigation_kind_counts"),
            (source_events, "source_event_types"),
            (source_actions, "source_action_types"),
        ]:
            counter.update({name: value for name, value in counter_items(activity.get(key))})
        file_name = safe_text(activity.get("dominant_file"))
        symbol = safe_text(activity.get("dominant_symbol"))
        if file_name:
            files.append(file_name)
        if symbol:
            symbols.append(symbol)

    result = dict(segment)
    result.update(
        {
            "file_extension_counts": dict(file_extensions),
            "file_role_counts": dict(file_roles),
            "command_family_counts": dict(command_families),
            "command_name_counts": dict(command_names),
            "command_category_counts": dict(command_categories),
            "symbol_kind_counts": dict(symbol_kinds),
            "debug_action_counts": dict(debug_actions),
            "vcs_action_counts": dict(vcs_actions),
            "navigation_kind_counts": dict(navigation_kinds),
            "source_event_types": dict(source_events),
            "source_action_types": dict(source_actions),
            "unique_file_count": len(set(files)),
            "unique_symbol_count": len(set(symbols)),
            "file_switch_count": sum(1 for left, right in zip(files, files[1:], strict=False) if left != right),
            "symbol_switch_count": sum(1 for left, right in zip(symbols, symbols[1:], strict=False) if left != right),
            "file_context_entropy": entropy_from_counts(Counter(files)),
            "symbol_context_entropy": entropy_from_counts(Counter(symbols)),
            "command_context_entropy": entropy_from_counts(command_families),
        }
    )
    return result


def enrich_segments_with_activity_context(segments: list[dict[str, Any]], activities: pd.DataFrame) -> list[dict[str, Any]]:
    """Attach aggregated context to segments that do not already carry it."""
    if not segments:
        return []
    if "file_extension_counts" in segments[0] and "command_family_counts" in segments[0]:
        return segments
    activity_lookup = {
        str(row["activity_id"]): row for row in activities.to_dict("records")
    }
    return [aggregate_segment_context(segment, activity_lookup) for segment in segments]


def segment_feature_dicts(segments: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Build broad sparse feature dictionaries for segment-level clustering."""
    result: list[dict[str, float]] = []
    for segment in segments:
        activities = [safe_text(value) for value in segment.get("activities", []) if safe_text(value)]
        bigrams = [f"{left}->{right}" for left, right in zip(activities, activities[1:], strict=False)]
        trigrams = [
            f"{activities[index]}->{activities[index + 1]}->{activities[index + 2]}"
            for index in range(max(0, len(activities) - 2))
        ]
        # Clustering features must be length-invariant. Raw magnitudes such as
        # log_activity_count/log_event_count/log_duration made HDBSCAN group
        # segments by size instead of behavior (clusters pinned at the
        # max_segment_activities cap); absolute sizes stay available on
        # segment_table for interpretation only.
        activity_total = max(1.0, float(segment.get("activity_count") or len(activities) or 1.0))
        item: dict[str, float] = {
            "num:unique_file_rate": float(segment.get("unique_file_count") or segment.get("file_count") or 0.0) / activity_total,
            "num:unique_symbol_rate": float(segment.get("unique_symbol_count") or 0.0) / activity_total,
            "num:file_switch_rate": float(segment.get("file_switch_count") or 0.0) / activity_total,
            "num:symbol_switch_rate": float(segment.get("symbol_switch_count") or 0.0) / activity_total,
            "num:file_context_entropy": float(segment.get("file_context_entropy") or 0.0),
            "num:symbol_context_entropy": float(segment.get("symbol_context_entropy") or 0.0),
            "num:command_context_entropy": float(segment.get("command_context_entropy") or 0.0),
            f"dominant_family={safe_text(segment.get('dominant_family')) or 'unknown'}": 1.0,
            f"dominant_activity={safe_text(segment.get('dominant_activity')) or 'unknown'}": 1.0,
        }
        item.update(normalized_counts(activities, "activity"))
        item.update(normalized_counts([activity_family(activity) for activity in activities], "family"))
        item.update(normalized_counts(bigrams, "bigram"))
        item.update(normalized_counts(trigrams, "trigram"))
        add_count_features(item, segment.get("file_extension_counts"), "file_ext")
        add_count_features(item, segment.get("file_role_counts"), "file_role")
        add_count_features(item, segment.get("command_family_counts"), "command_family")
        add_count_features(item, segment.get("command_name_counts"), "command_name")
        add_count_features(item, segment.get("command_category_counts"), "command_category")
        add_count_features(item, segment.get("symbol_kind_counts"), "symbol_kind")
        add_count_features(item, segment.get("debug_action_counts"), "debug_action")
        add_count_features(item, segment.get("vcs_action_counts"), "vcs_action")
        add_count_features(item, segment.get("navigation_kind_counts"), "navigation_kind")
        add_count_features(item, segment.get("source_event_types"), "source_event")
        add_count_features(item, segment.get("source_action_types"), "source_action")
        result.append(item)
    return result


def cluster_feature_rankings(
    feature_matrix: Any,
    feature_names: list[str],
    labels: np.ndarray,
    top_n: int = 12,
) -> dict[int, list[tuple[str, float]]]:
    """Rank features whose cluster mean most exceeds the global mean."""
    if len(labels) == 0 or feature_matrix.shape[0] == 0:
        return {}
    matrix = feature_matrix.tocsr() if sparse.issparse(feature_matrix) else sparse.csr_matrix(feature_matrix)
    global_mean = np.asarray(matrix.mean(axis=0)).ravel()
    rankings: dict[int, list[tuple[str, float]]] = {}
    for label in sorted(set(int(value) for value in labels if int(value) >= 0)):
        rows = np.where(labels == label)[0]
        if len(rows) == 0:
            continue
        cluster_mean = np.asarray(matrix[rows].mean(axis=0)).ravel()
        lift = cluster_mean - global_mean
        order = np.argsort(lift)[::-1]
        top: list[tuple[str, float]] = []
        for index in order:
            if lift[index] <= 0:
                break
            top.append((feature_names[index], float(lift[index])))
            if len(top) >= top_n:
                break
        rankings[label] = top
    return rankings
