import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


EVAL_TYPE_DICT = {
    "Perception": [
        "existence",
        "count",
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "OCR",
    ],
    "Cognition": [
        "commonsense_reasoning",
        "numerical_calculation",
        "text_translation",
        "code_reasoning",
    ],
}


def parse_args() -> argparse.Namespace:
    repo_root = Path("/home/srj/VLMsafeguard")
    code_dir = repo_root / "code"
    parser = argparse.ArgumentParser(
        description="Export and summarize MME soft-sweep runs into a single CSV/JSON."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=code_dir / "outputs/mme/soft_sweep",
        help="Root directory containing per-run generation outputs.",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=code_dir / "eval_results/mme/soft_sweep",
        help="Root directory to store eval_tool-format exports per run.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=code_dir / "eval_results/mme/soft_sweep_summary.csv",
        help="Path to write the aggregated CSV summary.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=code_dir / "eval_results/mme/soft_sweep_summary.json",
        help="Path to write the aggregated JSON summary.",
    )
    parser.add_argument(
        "--mm-jsonl",
        type=Path,
        default=code_dir / "data/mme_build/llava_mme_mm.jsonl",
        help="MME multimodal jsonl used for generation.",
    )
    parser.add_argument(
        "--mme-root",
        type=Path,
        default=Path("/s/datasets/MME/MME_Benchmark_release_version/MME_Benchmark"),
        help="Official MME benchmark root used by the exporter.",
    )
    parser.add_argument(
        "--export-script",
        type=Path,
        default=code_dir / "scripts/multimodal/export_mme_for_eval_tool.py",
        help="Exporter script path.",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Re-run exporter even if exported txt files already exist.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind}: {path}")


def iter_prediction_csvs(outputs_root: Path) -> Iterable[Path]:
    return sorted(outputs_root.glob("*/**/output_greedy.csv"))


def normalize_prediction(pred_ans: str) -> str:
    pred_ans = (pred_ans or "").strip().lower()
    if pred_ans in {"yes", "no"}:
        return pred_ans
    prefix = pred_ans[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


def divide_chunks(lines: list[str], n: int = 2) -> Iterable[list[str]]:
    for idx in range(0, len(lines), n):
        yield lines[idx : idx + n]


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_task_metrics(task_txt: Path) -> dict:
    lines = [line.strip() for line in task_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunk_lines = list(divide_chunks(lines, n=2))
    img_num = len(chunk_lines)
    if img_num == 0:
        return {
            "score": 0.0,
            "acc": 0.0,
            "acc_plus": 0.0,
            "total_questions": 0,
            "other_num": 0,
            "matched_questions": 0,
        }

    total_questions = 0
    other_num = 0
    matched_questions = 0
    acc_plus_correct_num = 0

    for img_items in chunk_lines:
        if len(img_items) != 2:
            raise ValueError(f"Expected 2 questions per image in {task_txt}, got {len(img_items)}")

        img_correct_num = 0
        for img_item in img_items:
            parts = img_item.split("\t")
            if len(parts) != 4:
                raise ValueError(f"Bad line in {task_txt}: {img_item!r}")
            _, _, gt_ans, pred_ans = parts
            gt_ans = gt_ans.strip().lower()
            pred_ans = normalize_prediction(pred_ans)

            if gt_ans not in {"yes", "no"}:
                raise ValueError(f"Unexpected GT answer in {task_txt}: {gt_ans!r}")

            total_questions += 1
            if pred_ans == gt_ans:
                matched_questions += 1
                img_correct_num += 1
            if pred_ans not in {"yes", "no"}:
                other_num += 1

        if img_correct_num == 2:
            acc_plus_correct_num += 1

    acc = safe_div(matched_questions, total_questions)
    acc_plus = safe_div(acc_plus_correct_num, img_num)
    score = (acc + acc_plus) * 100.0
    return {
        "score": score,
        "acc": acc,
        "acc_plus": acc_plus,
        "total_questions": total_questions,
        "other_num": other_num,
        "matched_questions": matched_questions,
    }


def parse_run_name(run_name: str) -> dict:
    prompt_stem, suffix = (run_name.split("__", 1) + [""])[:2] if "__" in run_name else (run_name, "")
    parsed = {
        "run_name": run_name,
        "prompt_stem": prompt_stem,
        "run_suffix": suffix,
        "lr": "",
        "epochs": "",
        "batch_size": "",
        "extra": "",
    }
    if not suffix:
        return parsed

    match = re.fullmatch(r"lr([^_]+)_ep(\d+)_bs(\d+)(?:_(.*))?", suffix)
    if match:
        parsed["lr"] = match.group(1)
        parsed["epochs"] = int(match.group(2))
        parsed["batch_size"] = int(match.group(3))
        parsed["extra"] = match.group(4) or ""
    else:
        parsed["extra"] = suffix
    return parsed


def export_if_needed(
    pred_csv: Path,
    export_dir: Path,
    args: argparse.Namespace,
) -> None:
    summary_path = export_dir / "summary.json"
    if summary_path.exists() and not args.force_export:
        return

    export_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(args.export_script),
            "--pred-csv",
            str(pred_csv),
            "--mm-jsonl",
            str(args.mm_jsonl),
            "--mme-root",
            str(args.mme_root),
            "--output-dir",
            str(export_dir),
        ],
        check=True,
    )


def summarize_run(run_name: str, pred_csv: Path, export_dir: Path) -> dict:
    parsed = parse_run_name(run_name)
    export_summary = json.loads((export_dir / "summary.json").read_text(encoding="utf-8"))

    result = {
        **parsed,
        "pred_csv": str(pred_csv.resolve()),
        "export_dir": str(export_dir.resolve()),
        "overall_accuracy": export_summary["overall"]["accuracy"],
        "overall_matched": export_summary["overall"]["matched"],
        "overall_total": export_summary["overall"]["total"],
        "question_mismatch_count": export_summary["question_mismatch_count"],
    }

    grand_total = 0.0
    for eval_type, tasks in EVAL_TYPE_DICT.items():
        eval_total = 0.0
        for task_name in tasks:
            metrics = compute_task_metrics(export_dir / f"{task_name}.txt")
            result[f"{task_name}_score"] = metrics["score"]
            result[f"{task_name}_acc"] = metrics["acc"]
            result[f"{task_name}_acc_plus"] = metrics["acc_plus"]
            eval_total += metrics["score"]
        result[f"{eval_type.lower()}_score"] = eval_total
        grand_total += eval_total

    result["mme_total_score"] = grand_total
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")

    fieldnames = [
        "run_name",
        "prompt_stem",
        "run_suffix",
        "lr",
        "epochs",
        "batch_size",
        "extra",
        "mme_total_score",
        "perception_score",
        "cognition_score",
        "overall_accuracy",
        "overall_matched",
        "overall_total",
        "question_mismatch_count",
        "pred_csv",
        "export_dir",
    ]
    for eval_type_tasks in EVAL_TYPE_DICT.values():
        for task_name in eval_type_tasks:
            fieldnames.extend(
                [
                    f"{task_name}_score",
                    f"{task_name}_acc",
                    f"{task_name}_acc_plus",
                ]
            )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_runs": len(rows),
        "best_run": rows[0] if rows else None,
        "runs": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_exists(args.outputs_root, "outputs root")
    ensure_exists(args.mm_jsonl, "mm_jsonl")
    ensure_exists(args.mme_root, "mme root")
    ensure_exists(args.export_script, "export script")

    pred_csvs = list(iter_prediction_csvs(args.outputs_root))
    if not pred_csvs:
        raise FileNotFoundError(f"No output_greedy.csv found under {args.outputs_root}")

    rows = []
    for pred_csv in pred_csvs:
        run_name = pred_csv.parents[1].name
        export_dir = args.eval_root / run_name
        export_if_needed(pred_csv=pred_csv, export_dir=export_dir, args=args)
        rows.append(summarize_run(run_name=run_name, pred_csv=pred_csv, export_dir=export_dir))

    rows.sort(key=lambda row: (-row["mme_total_score"], row["run_name"]))
    write_csv(rows, args.summary_csv)
    write_json(rows, args.summary_json)

    print(f"Wrote CSV summary : {args.summary_csv}")
    print(f"Wrote JSON summary: {args.summary_json}")
    print("Top runs:")
    for row in rows[:5]:
        print(
            f"  {row['run_name']}: total={row['mme_total_score']:.2f}, "
            f"perception={row['perception_score']:.2f}, cognition={row['cognition_score']:.2f}, "
            f"acc={row['overall_accuracy']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
