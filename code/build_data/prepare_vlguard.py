#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "items", "records", "samples"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def load_rows_with_source(raw_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    rows: List[Tuple[Path, Dict[str, Any]]] = []
    files = sorted(list(raw_dir.rglob("*.json")) + list(raw_dir.rglob("*.jsonl")))
    if not files:
        raise RuntimeError(f"No json/jsonl found in {raw_dir}")

    for path in files:
        for row in load_json_or_jsonl(path):
            rows.append((path, row))
    return rows


def extract_samples(rows_with_source: List[Tuple[Path, Dict[str, Any]]], image_root: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    for row_idx, (source_path, row) in enumerate(rows_with_source):
        rid = str(row.get("id", f"sample_{row_idx}"))
        rel_img = row.get("image")
        if not rel_img:
            continue

        img_path = str((image_root / rel_img).resolve())
        safe_flag = row.get("safe", None)
        turns = row.get("instr-resp", [])
        if not isinstance(turns, list) or len(turns) == 0:
            continue

        for turn_idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue

            question = None
            label = None
            label_source = None

            if turn.get("safe_instruction"):
                question = str(turn["safe_instruction"]).strip()
                label = 1
                label_source = "safe_instruction"
            elif turn.get("unsafe_instruction"):
                question = str(turn["unsafe_instruction"]).strip()
                label = 0
                label_source = "unsafe_instruction"
            elif turn.get("instruction"):
                question = str(turn["instruction"]).strip()
                if isinstance(safe_flag, bool):
                    label = 1 if safe_flag else 0
                    label_source = "sample_safe_flag"

            if not question or label is None:
                continue

            samples.append({
                "id": f"{rid}_{turn_idx}",
                "raw_id": rid,
                "source_file": str(source_path),
                "rel_image_path": str(rel_img),
                "image_path": img_path,
                "question": question,
                "label": int(label),
                "label_source": label_source,
            })

    return samples


def dedupe_samples(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    deduped = []
    seen = set()
    duplicates_removed = 0

    for sample in samples:
        key = (sample["image_path"], sample["question"], sample["label"])
        if key in seen:
            duplicates_removed += 1
            continue
        seen.add(key)
        deduped.append(sample)

    return deduped, {"duplicates_removed": duplicates_removed}


def find_conflicting_labels(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    key_to_labels: Dict[Tuple[str, str], set] = defaultdict(set)
    key_to_examples: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        key = (sample["image_path"], sample["question"])
        key_to_labels[key].add(sample["label"])
        if len(key_to_examples[key]) < 2:
            key_to_examples[key].append(sample)

    conflicts = []
    for key, labels in key_to_labels.items():
        if len(labels) > 1:
            conflicts.append({
                "image_path": key[0],
                "question": key[1],
                "labels": sorted(labels),
                "examples": key_to_examples[key],
            })
    return conflicts


def audit_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    label_counts = Counter(sample["label"] for sample in samples)
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    question_counts = Counter(sample["question"] for sample in samples)

    for sample in samples:
        by_group[sample["image_path"]].append(sample)

    mixed_groups = 0
    mixed_rows = 0
    safe_on_mixed = 0
    unsafe_on_mixed = 0
    for group_samples in by_group.values():
        labels = {sample["label"] for sample in group_samples}
        if labels == {0, 1}:
            mixed_groups += 1
            mixed_rows += len(group_samples)
            safe_on_mixed += sum(1 for sample in group_samples if sample["label"] == 1)
            unsafe_on_mixed += sum(1 for sample in group_samples if sample["label"] == 0)

    return {
        "total_samples": len(samples),
        "total_groups": len(by_group),
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "mixed_label_groups": mixed_groups,
        "mixed_label_rows": mixed_rows,
        "safe_rows_on_mixed_groups": safe_on_mixed,
        "unsafe_rows_on_mixed_groups": unsafe_on_mixed,
        "duplicate_questions": int(sum(1 for count in question_counts.values() if count > 1)),
    }


def build_group_records(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[sample["image_path"]].append(sample)

    group_records = []
    for group_key, group_samples in groups.items():
        label_counts = Counter(sample["label"] for sample in group_samples)
        group_records.append({
            "group_key": group_key,
            "samples": group_samples,
            "total": len(group_samples),
            "label_0": label_counts.get(0, 0),
            "label_1": label_counts.get(1, 0),
        })
    return group_records


def stratified_group_split(samples: List[Dict[str, Any]], train_ratio: float, seed: int):
    group_records = build_group_records(samples)
    rng = random.Random(seed)
    profile_buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for record in group_records:
        profile_buckets[(record["label_0"], record["label_1"])].append(record)

    train_groups = []
    test_groups = []
    for profile in sorted(profile_buckets):
        bucket = profile_buckets[profile]
        rng.shuffle(bucket)
        n_train = round(len(bucket) * train_ratio)
        train_groups.extend(bucket[:n_train])
        test_groups.extend(bucket[n_train:])

    train_samples = [sample for record in train_groups for sample in record["samples"]]
    test_samples = [sample for record in test_groups for sample in record["samples"]]
    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def balance_train_set(train_samples: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    safe = [s for s in train_samples if s['label'] == 1]
    unsafe = [s for s in train_samples if s['label'] == 0]
    rng.shuffle(unsafe)
    num_safe = len(safe)
    balanced_unsafe = unsafe[:num_safe]
    balanced_train = safe + balanced_unsafe
    rng.shuffle(balanced_train)
    return balanced_train


def split_summary(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    label_counts = Counter(row["label"] for row in rows)
    return {
        "total": len(rows),
        "unsafe": int(label_counts.get(0, 0)),
        "safe": int(label_counts.get(1, 0)),
        "groups": len({row["image_path"] for row in rows}),
    }


def balanced_label_truncate(samples: List[Dict[str, Any]], max_label0: Optional[int], max_label1: Optional[int], seed: int):
    rng = random.Random(seed)
    grouped = {0: [], 1: []}
    for sample in samples:
        grouped[sample["label"]].append(sample)

    for label in (0, 1):
        rng.shuffle(grouped[label])

    limit0 = max_label0 if max_label0 is not None else len(grouped[0])
    limit1 = max_label1 if max_label1 is not None else len(grouped[1])

    train0 = grouped[0][:limit0]
    test0 = grouped[0][limit0:]
    train1 = grouped[1][:limit1]
    test1 = grouped[1][limit1:]

    train_samples = train0 + train1
    test_samples = test0 + test1
    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            row_to_write = {
                "id": row["id"],
                "image_path": row["image_path"],
                "question": row["question"],
                "label": row["label"],
            }
            f.write(json.dumps(row_to_write, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--max_train_label0", type=int, default=None,
                        help="If set, keep at most this many label=0 samples in train; rest goes to test")
    parser.add_argument("--max_train_label1", type=int, default=None,
                        help="If set, keep at most this many label=1 samples in train; rest goes to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be in (0, 1)")

    raw_dir = Path(args.raw_dir)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)

    rows_with_source = load_rows_with_source(raw_dir)
    samples = extract_samples(rows_with_source, image_root)
    if not samples:
        raise RuntimeError("No samples parsed. Please check field names in raw data.")

    samples, dedupe_info = dedupe_samples(samples)
    conflicts = find_conflicting_labels(samples)
    if conflicts:
        raise RuntimeError(
            f"Found {len(conflicts)} conflicting label assignments for identical image/question pairs. "
            "Please inspect the raw dataset before splitting."
        )

    audit = audit_samples(samples)
    audit.update(dedupe_info)
    audit["conflicting_pairs"] = len(conflicts)

    if args.max_train_label0 is not None or args.max_train_label1 is not None:
        train_samples, test_samples = balanced_label_truncate(
            samples,
            max_label0=args.max_train_label0,
            max_label1=args.max_train_label1,
            seed=args.seed,
        )
    else:
        train_samples, test_samples = stratified_group_split(samples, args.train_ratio, args.seed)
        # 平衡训练集以达到 safe:unsafe = 1:1
        train_samples = balance_train_set(train_samples, args.seed)

    split_stats = {
        "train": split_summary(train_samples),
        "test": split_summary(test_samples),
    }

    train_groups = {row["image_path"] for row in train_samples}
    test_groups = {row["image_path"] for row in test_samples}
    split_stats["group_overlap"] = len(train_groups & test_groups)
    split_stats["actual_train_ratio"] = len(train_samples) / len(samples)
    split_stats["actual_test_ratio"] = len(test_samples) / len(samples)

    report = {
        "audit": audit,
        "split": split_stats,
        "config": {
            "raw_dir": str(raw_dir),
            "image_root": str(image_root),
            "out_dir": str(out_dir),
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "audit_only": args.audit_only,
            "split_strategy": "grouped_by_image_path_profile_bucket_split",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if args.audit_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("Audit only: no dataset files were written.")
        return

    write_jsonl(out_dir / "train_mm.jsonl", train_samples)
    write_jsonl(out_dir / "test_mm.jsonl", test_samples)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Done.")
    print("Output:", out_dir.resolve())


if __name__ == "__main__":
    main()
