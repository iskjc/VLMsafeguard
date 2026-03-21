#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`([{<\s]+", "", text)
    text = re.sub(r"[\"'`)\]}>.,;:!?。\s]+$", "", text)
    return text


def normalize_prediction(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    if first_line.startswith("yes"):
        return "yes"
    if first_line.startswith("no"):
        return "no"
    return first_line


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad json on line {line_no} in {path}: {exc}") from exc
    return rows


def read_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_answers(row: dict) -> list[str]:
    for key in ["answer", "answers", "gt_answer", "ground_truth", "target", "annotation", "caption", "captions"]:
        if key not in row or row[key] in (None, ""):
            continue
        value = row[key]
        if isinstance(value, list):
            return [normalize_text(v) for v in value if normalize_text(v)]
        return [normalize_text(value)]
    raise KeyError(f"Row is missing answer-like field: {sorted(row.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", type=Path, required=True)
    parser.add_argument("--mm-jsonl", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--category-field", type=str, default="category")
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    pred_rows = read_predictions(args.pred_csv)
    mm_rows = read_jsonl(args.mm_jsonl)

    if len(pred_rows) != len(mm_rows):
        raise ValueError(f"Row count mismatch: pred={len(pred_rows)}, mm_jsonl={len(mm_rows)}")

    total = 0
    matched = 0
    by_category = defaultdict(lambda: {"total": 0, "matched": 0})
    examples = []
    pred_counter = Counter()

    for idx, (pred, row) in enumerate(zip(pred_rows, mm_rows), start=1):
        answers = get_answers(row)
        prediction = normalize_prediction(pred.get("output", ""))
        pred_counter[prediction] += 1
        category = str(row.get(args.category_field, "default"))
        is_match = prediction in answers

        total += 1
        by_category[category]["total"] += 1
        if is_match:
            matched += 1
            by_category[category]["matched"] += 1
        elif len(examples) < args.examples:
            examples.append(
                {
                    "row_idx": idx,
                    "id": row.get("id", idx),
                    "category": category,
                    "question": row.get("question"),
                    "answers": answers,
                    "prediction": prediction,
                }
            )

    result = {
        "pred_csv": str(args.pred_csv.resolve()),
        "mm_jsonl": str(args.mm_jsonl.resolve()),
        "overall": {
            "total": total,
            "matched": matched,
            "accuracy": matched / total if total else 0.0,
        },
        "by_category": {
            category: {
                "total": stats["total"],
                "matched": stats["matched"],
                "accuracy": stats["matched"] / stats["total"] if stats["total"] else 0.0,
            }
            for category, stats in sorted(by_category.items())
        },
        "prediction_distribution_top20": pred_counter.most_common(20),
        "mismatch_examples": examples,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
