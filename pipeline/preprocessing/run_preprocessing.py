"""Convert KaVE ZIP logs into canonical event and session JSONL.

From the repository root, run python -m pipeline.preprocessing.run_preprocessing.
Paths default to configs/paths.json. Use --max-zips to limit the input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the repository modules available when this file is invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from project_paths import get_paths

from pipeline.preprocessing.canonical_event import CanonicalEvent
from pipeline.preprocessing.kave_parser import KaVEParser
from pipeline.preprocessing.kave_normalizer import KaVENormalizer
from pipeline.preprocessing.session_segmenter import SessionSegmenter


# CLI argument parsing

def parse_args() -> argparse.Namespace:
    paths = get_paths()
    parser = argparse.ArgumentParser(
        description="PHASE_1: Parse KaVE dataset -> CanonicalEvent -> Session JSONL",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(paths.raw_data),
        help="Raw dataset directory (default: configs/paths.json raw_data).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(paths.preprocessing),
        help="Intermediate output directory (default: configured intermediate/preprocessing).",
    )
    parser.add_argument(
        "--max-zips",
        type=int,
        default=None,
        help="Limit processing to the first N ZIP files (for quick testing).",
    )
    parser.add_argument(
        "--inactivity-min",
        type=float,
        default=15.0,
        help="Inactivity gap threshold in minutes for session splitting (default: 15).",
    )
    parser.add_argument(
        "--min-session-events",
        type=int,
        default=10,
        help="Minimum events per session; shorter sessions are discarded (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="Number of events to buffer before flushing to disk (default: 100000).",
    )
    return parser.parse_args()


# Pipeline execution

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"[ERROR] Dataset root not found: {dataset_root}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file paths
    events_jsonl = output_dir / "canonical_events.jsonl"
    sessions_jsonl = output_dir / "sessions.jsonl"
    stats_json = output_dir / "preprocessing_stats.json"

    # ------------------------------------------------------------------
    # Initialize components
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE_1: KaVE Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"  Dataset root:    {dataset_root}")
    print(f"  Output directory: {output_dir}")
    print(f"  Inactivity gap:  {args.inactivity_min} min")
    print(f"  Min session len: {args.min_session_events} events")
    if args.max_zips:
        print(f"  Max ZIP files:   {args.max_zips} (testing mode)")
    print()

    parser = KaVEParser(dataset_root)
    normalizer = KaVENormalizer()
    segmenter = SessionSegmenter(
        inactivity_threshold_min=args.inactivity_min,
        min_session_events=args.min_session_events,
    )

    # ------------------------------------------------------------------
    # Discover ZIP files
    # ------------------------------------------------------------------
    print("[1/3] Discovering ZIP files...")
    t0 = time.time()
    zip_paths = parser.discover_zip_files()
    print(f"       Found {len(zip_paths)} ZIP archives.")

    if args.max_zips:
        zip_paths = zip_paths[: args.max_zips]
        print(f"       Limited to {len(zip_paths)} ZIPs for testing.")

    # ------------------------------------------------------------------
    # Parse + Normalize -> CanonicalEvents (streaming with buffered writes)
    # ------------------------------------------------------------------
    print("[2/3] Parsing & normalizing events...")
    t1 = time.time()

    total_parsed = 0
    total_normalized = 0
    parse_errors = 0
    normalize_errors = 0

    # Session segmentation needs the complete event list.
    all_events: list[CanonicalEvent] = []

    # Write canonical events in batches.
    with open(events_jsonl, "w", encoding="utf-8") as f_events:
        for raw_event in parser.iter_events(zip_paths=zip_paths):
            total_parsed += 1

            # Normalize: KaVE raw -> CanonicalEvent
            try:
                canonical = normalizer.normalize(raw_event)
                total_normalized += 1
                all_events.append(canonical)

                # Write to JSONL in batches for memory efficiency
                if len(all_events) % args.batch_size == 0:
                    for evt in all_events[-args.batch_size :]:
                        f_events.write(json.dumps(evt.to_dict(), ensure_ascii=False) + "\n")
                    f_events.flush()
                    elapsed = time.time() - t1
                    rate = total_parsed / elapsed if elapsed > 0 else 0
                    print(f"       Processed {total_parsed:>10,} events  "
                          f"({rate:,.0f} evt/s)  "
                          f"[{len(all_events):,} normalized]", end="\r")
            except Exception as exc:
                normalize_errors += 1
                if normalize_errors <= 5:
                    print(f"\n[WARN] Normalization error #{normalize_errors}: {exc}")

            # Progress indicator every 50K events
            if total_parsed % 50000 == 0 and total_parsed > 0:
                elapsed = time.time() - t1
                rate = total_parsed / elapsed if elapsed > 0 else 0
                print(f"       Processed {total_parsed:>10,} events  "
                      f"({rate:,.0f} evt/s)  "
                      f"[{total_normalized:,} normalized]")

        # Flush remaining events in the last partial batch
        remaining_start = (len(all_events) // args.batch_size) * args.batch_size
        for evt in all_events[remaining_start:]:
            f_events.write(json.dumps(evt.to_dict(), ensure_ascii=False) + "\n")
        f_events.flush()

    t2 = time.time()
    parse_duration = t2 - t1
    print(f"\n       Done. {total_parsed:,} parsed, "
          f"{total_normalized:,} normalized, "
          f"{parse_errors} parse errors, "
          f"{normalize_errors} normalize errors.")
    print(f"       Duration: {parse_duration:.1f}s "
          f"({total_parsed / parse_duration:,.0f} evt/s)")
    print(f"       Events written to: {events_jsonl}")

    # ------------------------------------------------------------------
    # Session segmentation
    # ------------------------------------------------------------------
    print("[3/3] Segmenting into sessions...")
    t3 = time.time()

    # Sort events by (session_id, timestamp) before segmentation
    sorted_events = segmenter.sort_events(all_events)
    sessions = segmenter.segment(sorted_events)

    # Write sessions to JSONL (each line = one session, array of event dicts)
    with open(sessions_jsonl, "w", encoding="utf-8") as f_sessions:
        for session in sessions:
            session_data = [evt.to_dict() for evt in session]
            f_sessions.write(json.dumps(session_data, ensure_ascii=False) + "\n")

    t4 = time.time()
    seg_duration = t4 - t3
    print(f"       {len(sessions):,} sessions extracted "
          f"from {len(all_events):,} events.")
    print(f"       Duration: {seg_duration:.1f}s")

    # ------------------------------------------------------------------
    # Compute & save statistics
    # ------------------------------------------------------------------
    session_stats = segmenter.get_session_stats(sessions)
    total_duration = time.time() - t0

    stats = {
        "pipeline": "PHASE_1_preprocessing",
        "dataset_root": str(dataset_root),
        "total_zips_discovered": len(parser.discover_zip_files()),
        "zips_processed": len(zip_paths),
        "total_parsed": total_parsed,
        "total_normalized": total_normalized,
        "parse_errors": parse_errors,
        "normalize_errors": normalize_errors,
        "session_stats": session_stats,
        "output_files": {
            "canonical_events_jsonl": str(events_jsonl),
            "sessions_jsonl": str(sessions_jsonl),
            "stats_json": str(stats_json),
        },
        "timing": {
            "parse_normalize_seconds": round(parse_duration, 1),
            "segmentation_seconds": round(seg_duration, 1),
            "total_seconds": round(total_duration, 1),
        },
        "config": {
            "inactivity_threshold_min": args.inactivity_min,
            "min_session_events": args.min_session_events,
        },
    }

    with open(stats_json, "w", encoding="utf-8") as f_stats:
        json.dump(stats, f_stats, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("PHASE_1 Complete!")
    print("=" * 60)
    print(f"  Total events:      {total_normalized:>10,}")
    print(f"  Total sessions:    {len(sessions):>10,}")
    print(f"  Avg events/session: {session_stats['avg_events_per_session']:>10.1f}")
    print(f"  Avg duration (min): {session_stats['avg_duration_minutes']:>10.1f}")
    print(f"  Total time:        {total_duration:>10.1f}s")
    print()
    print(f"  Outputs:")
    print(f"    Events:   {events_jsonl}")
    print(f"    Sessions: {sessions_jsonl}")
    print(f"    Stats:    {stats_json}")
    print()


if __name__ == "__main__":
    main()

