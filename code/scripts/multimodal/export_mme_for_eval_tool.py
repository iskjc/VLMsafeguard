import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

QUESTION_SUFFIXES = [
    "Answer the question using a single word or phrase.",
    "Please answer yes or no.",
    "Please answer with yes or no.",
    "Answer yes or no.",
    "Please answer no or yes.",
]

ANSWER_DIR_CANDIDATES = [
    "questions_answers_YN",
    "questions_answers",
    "question_answers_YN",
    "question_answers",
    "",
]

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def strip_answer_suffix(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").strip()
    text = text.replace("\n", " ")
    text = normalize_space(text)

    changed = True
    while changed:
        changed = False
        for suffix in QUESTION_SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)].strip()
                text = normalize_space(text)
                changed = True

    return text


def canonicalize_question(text: str) -> str:
    text = strip_answer_suffix(text)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_yes_no(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    first_line = text.splitlines()[0].strip()
    first_line = re.sub(r"^[\"'`([{<\s]+", "", first_line)
    first_line = re.sub(r"[\"'`)\]}>.,;:!?。\s]+$", "", first_line)
    lowered = first_line.casefold()

    if re.match(r"^(yes|yeah|yep|y|true)\b", lowered):
        return "yes"
    if re.match(r"^(no|nope|n|false)\b", lowered):
        return "no"
    return first_line


def extract_question_from_input(input_text: str) -> str:
    text = (input_text or "").replace("\r\n", "\n")
    text = text.replace("USER: <image>", "", 1).strip()
    if text.endswith("ASSISTANT:"):
        text = text[: -len("ASSISTANT:")].strip()
    return text


def read_mm_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_idx} of {path}: {exc}") from exc
    return rows


def read_predictions_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in prediction CSV: {path}")

    return rows


def answer_file_candidates(mme_root: Path, category: str, question_id: str, image_path: str) -> list[Path]:
    image_name = Path(image_path).name
    image_stem = Path(image_path).stem
    qid_name = Path(question_id).name
    qid_stem = Path(question_id).stem
    category_dir = mme_root / category

    names = [
        f"{image_stem}.txt",
        f"{image_name}.txt",
        f"{qid_stem}.txt",
        f"{qid_name}.txt",
    ]

    candidates = []
    for subdir in ANSWER_DIR_CANDIDATES:
        base = category_dir / subdir if subdir else category_dir
        for name in names:
            candidates.append(base / name)
    return candidates


def find_answer_file(mme_root: Path, meta: dict) -> Path:
    category = str(meta["category"])
    question_id = str(meta["question_id"])
    image_path = str(meta["image_path"])

    candidates = answer_file_candidates(mme_root, category, question_id, image_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find answer file for question_id={question_id}\nTried:\n{tried}"
    )


def parse_answer_file(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split("\t")]
            parts = [p for p in parts if p != ""]

            if len(parts) >= 2:
                question = parts[0]
                answer = parts[-1]
                pairs.append((question, answer))
                continue

            match = re.match(r"^(.*?)(yes|no)\s*$", line, flags=re.IGNORECASE)
            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                pairs.append((question, answer))
                continue

            raise ValueError(f"Unrecognized line format in answer file {path}: {raw_line!r}")

    if not pairs:
        raise ValueError(f"No QA pairs found in {path}")

    return pairs

def build_answer_entries(path: Path) -> list[dict]:
    entries = []
    for question, answer in parse_answer_file(path):
        entries.append(
            {
                "question": question,
                "question_key": canonicalize_question(question),
                "answer": normalize_yes_no(answer) or normalize_space(answer),
            }
        )
    return entries


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export MME predictions into eval_tool-style category txt files."
    )
    parser.add_argument(
        "--pred-csv",
        type=Path,
        default=Path("code/outputs/mme/soft_all_default/llava-1.5-7b-hf_with_soft_all_default_llava_mme/output_greedy.csv"),
        help="Path to model prediction CSV.",
    )
    parser.add_argument(
        "--mm-jsonl",
        type=Path,
        default=Path("code/data/mme_build/llava_mme_mm.jsonl"),
        help="Path to the MME mm_jsonl used for generation.",
    )
    parser.add_argument(
        "--mme-root",
        type=Path,
        default=Path("/s/datasets/MME/MME_Benchmark_release_version/MME_Benchmark"),
        help="Root of the official MME benchmark data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("code/eval_results/mme/soft_for_eval_tool"),
        help="Directory to write exported category txt files.",
    )
    parser.add_argument(
        "--strict-question-match",
        action="store_true",
        help="Fail immediately if CSV question text and mm_jsonl question text differ.",
    )
    args = parser.parse_args()

    pred_rows = read_predictions_csv(args.pred_csv)
    mm_rows = read_mm_jsonl(args.mm_jsonl)

    if len(pred_rows) != len(mm_rows):
        raise ValueError(
            f"Row count mismatch: pred_csv has {len(pred_rows)} rows, mm_jsonl has {len(mm_rows)} rows"
        )

    ensure_dir(args.output_dir)

    answer_cache: dict[Path, list[dict]] = {}
    qid_seen_count = defaultdict(int)
    export_lines: dict[str, list[str]] = defaultdict(list)
    summary: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "matched": 0})
    question_mismatches = []

    for idx, (pred, meta) in enumerate(zip(pred_rows, mm_rows), start=1):
        csv_question = pred.get("prompt") or extract_question_from_input(pred.get("input", ""))
        mm_question = meta["question"]

        csv_q_key = canonicalize_question(csv_question)
        mm_q_key = canonicalize_question(mm_question)

        if csv_q_key != mm_q_key:
            mismatch = {
                "row_idx": idx,
                "csv_question": strip_answer_suffix(csv_question),
                "mm_question": strip_answer_suffix(mm_question),
                "question_id": meta.get("question_id"),
            }
            question_mismatches.append(mismatch)
            if args.strict_question_match:
                raise ValueError(
                    f"Question mismatch at row {idx}: {mismatch['csv_question']!r} != {mismatch['mm_question']!r}"
                )

        answer_file = find_answer_file(args.mme_root, meta)
        if answer_file not in answer_cache:
            answer_cache[answer_file] = build_answer_entries(answer_file)

        entries = answer_cache[answer_file]
        question_id = str(meta["question_id"])
        local_idx = qid_seen_count[question_id]
        qid_seen_count[question_id] += 1

        matched_entries = [e for e in entries if e["question_key"] == mm_q_key]
        if len(matched_entries) == 1:
            gt_answer = matched_entries[0]["answer"]
        else:
            if local_idx >= len(entries):
                raise IndexError(
                    f"Fallback index out of range for row {idx}, question_id={question_id}, "
                    f"answer_file={answer_file}, local_idx={local_idx}, entries={len(entries)}"
                )
            gt_answer = entries[local_idx]["answer"]


        pred_answer = normalize_yes_no(pred.get("output", ""))
        category = str(meta["category"])
        image_name = Path(str(meta["image_path"])).name
        export_question = strip_answer_suffix(mm_question)

        export_lines[category].append(
            f"{image_name}\t{export_question}\t{gt_answer}\t{pred_answer}"
        )

        summary[category]["total"] += 1
        if pred_answer.casefold() == gt_answer.casefold():
            summary[category]["matched"] += 1

    for category, lines in export_lines.items():
        out_file = args.output_dir / f"{category}.txt"
        with out_file.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    overall_total = sum(v["total"] for v in summary.values())
    overall_matched = sum(v["matched"] for v in summary.values())

    summary_json = {
        "pred_csv": str(args.pred_csv.resolve()),
        "mm_jsonl": str(args.mm_jsonl.resolve()),
        "mme_root": str(args.mme_root),
        "output_dir": str(args.output_dir.resolve()),
        "overall": {
            "total": overall_total,
            "matched": overall_matched,
            "accuracy": (overall_matched / overall_total) if overall_total else 0.0,
        },
        "by_category": {
            k: {
                "total": v["total"],
                "matched": v["matched"],
                "accuracy": (v["matched"] / v["total"]) if v["total"] else 0.0,
            }
            for k, v in sorted(summary.items())
        },
        "question_mismatches": question_mismatches[:20],
        "question_mismatch_count": len(question_mismatches),
    }

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f"Exported eval files to: {args.output_dir}")
    print(f"Summary written to   : {summary_path}")
    print(f"Overall matched      : {overall_matched}/{overall_total}")
    if question_mismatches:
        print(f"Question mismatches  : {len(question_mismatches)} (see summary.json)")
    print()
    print("Next step:")
    print("  Run MME eval_tool with --results_dir pointing at this output directory.")
    print("  For example:")
    print("    cd /s/datasets/MME/eval_tool")
    print(f"    python3 calculation.py --results_dir {args.output_dir.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
