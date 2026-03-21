import argparse
import json
from pathlib import Path


def load_rows(path: Path):
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
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

    raise ValueError(f"Unsupported input format: {path}")


def load_answers(path: Path | None):
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    answer_map = {}

    if isinstance(data, list):
        for row in data:
            qid = row.get("questionId") or row.get("question_id") or row.get("id")
            ans = row.get("prediction") or row.get("answer")
            if qid is not None and ans is not None:
                answer_map[str(qid)] = ans
    elif isinstance(data, dict):
        for k, v in data.items():
            answer_map[str(k)] = v
    else:
        raise ValueError(f"Unsupported answers format: {path}")

    return answer_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to original GQA json/jsonl file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to converted output jsonl",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="/s/datasets/gqa/images/images",
        help="Root directory containing GQA images",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        choices=[0, 1],
        help="Label to assign to all rows; harmless GQA should use 1",
    )
    parser.add_argument(
        "--answers-json",
        type=str,
        default=None,
        help="Optional GQA answers/predictions json file to attach as 'answer'",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip rows whose images do not exist instead of raising an error",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    image_root = Path(args.image_root)
    answers_path = Path(args.answers_json) if args.answers_json else None

    rows = load_rows(input_path)
    answer_map = load_answers(answers_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows):
            total += 1

            qid = row.get("question_id") or row.get("questionId") or row.get("id")
            image_name = row.get("image") or row.get("image_name") or row.get("img")
            question = row.get("question") or row.get("text")

            if qid is None:
                qid = f"sample_{idx}"

            if not image_name or not question:
                skipped += 1
                continue

            image_path = (image_root / str(image_name)).resolve()

            if not image_path.exists():
                if args.skip_missing_images:
                    skipped += 1
                    continue
                raise FileNotFoundError(f"Missing image: {image_path}")

            record = {
                "id": str(qid),
                "question_id": str(qid),
                "image_path": str(image_path),
                "question": str(question).strip(),
                "label": int(args.label),
            }

            if "category" in row:
                record["category"] = row["category"]

            if str(qid) in answer_map:
                record["answer"] = answer_map[str(qid)]

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Input rows   : {total}")
    print(f"Written rows : {written}")
    print(f"Skipped rows : {skipped}")
    print(f"Output file  : {output_path}")


if __name__ == "__main__":
    main()
