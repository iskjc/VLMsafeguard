import argparse
import json
from pathlib import Path


def load_rows(path: Path):
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

    raise ValueError(f"Unsupported input format: {path}")


def make_unique_id(question_id: str, idx: int) -> str:
    safe_qid = question_id.replace("/", "__")
    return f"{safe_qid}__{idx}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to original MME json/jsonl file",
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
        default="/s/datasets/MME/MME_Benchmark_release_version/MME_Benchmark",
        help="Root directory of MME benchmark images",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        choices=[0, 1],
        help="Label assigned to all samples; harmless benchmark usually uses 1",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip rows with missing images instead of raising an error",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    image_root = Path(args.image_root)

    rows = load_rows(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows):
            total += 1

            question_id = row.get("question_id") or row.get("questionId") or row.get("id")
            image_rel = row.get("image")
            question = row.get("question") or row.get("text")
            category = row.get("category")

            if not question_id or not image_rel or not question:
                skipped += 1
                continue

            image_path = (image_root / str(image_rel)).resolve()

            if not image_path.exists():
                if args.skip_missing_images:
                    skipped += 1
                    continue
                raise FileNotFoundError(f"Missing image: {image_path}")

            record = {
                "id": make_unique_id(str(question_id), idx),
                "question_id": str(question_id),
                "image_path": str(image_path),
                "question": str(question).strip(),
                "label": int(args.label),
            }

            if category is not None:
                record["category"] = category

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Input rows   : {total}")
    print(f"Written rows : {written}")
    print(f"Skipped rows : {skipped}")
    print(f"Output file  : {output_path}")


if __name__ == "__main__":
    main()
