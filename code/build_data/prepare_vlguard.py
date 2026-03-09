#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        for k in ["data", "items", "records", "samples"]:
            if k in data and isinstance(data[k], list):
                return data[k]
    return []


def pick_first(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in [None, ""]:
            return d[k]
    return None


def normalize_label(v: Any) -> Optional[int]:
    # 0=unsafe, 1=safe
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)) and int(v) in [0, 1]:
        return int(v)
    s = str(v).strip().lower()
    if s in {"safe", "harmless", "benign", "clean", "1", "true"}:
        return 1
    if s in {"unsafe", "harmful", "toxic", "jailbreak", "0", "false"}:
        return 0
    if "safe" in s:
        return 1
    if "unsafe" in s or "harm" in s:
        return 0
    return None

def resolve_image_path(image_root: Path, p: str) -> str:
    pp = Path(str(p))
    return str(pp.resolve()) if pp.is_absolute() else str((image_root / pp).resolve())


def extract_samples(rows: List[Dict[str, Any]], image_root: Path) -> List[Dict[str, Any]]:
    """
    适配 VLMsafeguard:
    {
      "id": ...,
      "image": "bad_ads/xxx.png",
      "safe": True/False,
      "instr-resp": [
        {"instruction": "...", "response": "..."} 或
        {"safe_instruction": "...", "response": "..."} 或
        {"unsafe_instruction": "...", "response": "..."}
      ]
    }

    统一输出:
    label: 1=safe, 0=unsafe
    """
    samples: List[Dict[str, Any]] = []

    for i, r in enumerate(rows):
        rid = str(r.get("id", f"sample_{i}"))
        rel_img = r.get("image")
        if not rel_img:
            continue

        img_path = (image_root / rel_img).resolve()
        safe_flag = r.get("safe", None)  # bool
        turns = r.get("instr-resp", [])

        if not isinstance(turns, list) or len(turns) == 0:
            continue

        for j, t in enumerate(turns):
            if not isinstance(t, dict):
                continue

            prompt = None
            label = None

            # 显式字段优先
            if "safe_instruction" in t and t["safe_instruction"]:
                prompt = str(t["safe_instruction"]).strip()
                label = 1
            elif "unsafe_instruction" in t and t["unsafe_instruction"]:
                prompt = str(t["unsafe_instruction"]).strip()
                label = 0
            elif "instruction" in t and t["instruction"]:
                prompt = str(t["instruction"]).strip()
                # 回退到样本级 safe 标记
                if isinstance(safe_flag, bool):
                    label = 1 if safe_flag else 0

            if not prompt or label is None:
                continue

            samples.append({
                "id": f"{rid}_{j}",
                "image_path": str(img_path),
                "prompt": prompt,
                "label": label,
            })

    return samples


def stratified_split(samples: List[Dict[str, Any]], train_ratio: float, seed: int):
    by_label = {0: [], 1: []}
    for s in samples:
        by_label[s["label"]].append(s)

    rng = random.Random(seed)
    for k in by_label:
        rng.shuffle(by_label[k])

    train, test = [], []
    for k in [0, 1]:
        n = len(by_label[k])
        n_train = int(n * train_ratio)
        train.extend(by_label[k][:n_train])
        test.extend(by_label[k][n_train:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_txt(path: Path, lines: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in lines:
            f.write(x.strip() + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True)        # e.g. data/data_vlguard/raw
    ap.add_argument("--image_root", type=str, required=True)     # e.g. data/data_vlguard/images
    ap.add_argument("--out_dir", type=str, required=True)        # e.g. data/data_vlguard/processed
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)

    files = sorted(list(raw_dir.rglob("*.json")) + list(raw_dir.rglob("*.jsonl")))
    if not files:
        raise RuntimeError(f"No json/jsonl found in {raw_dir}")

    rows = []
    for p in files:
        rows.extend(load_json_or_jsonl(p))

    samples = extract_samples(rows, image_root)
    if not samples:
        raise RuntimeError("No samples parsed. Please check field names in raw data.")

    train_samples, test_samples = stratified_split(samples, args.train_ratio, args.seed)

    # 多模态格式（你后续训练直接用）
    write_jsonl(out_dir / "train_mm.jsonl", train_samples)
    write_jsonl(out_dir / "test_mm.jsonl", test_samples)

    # 文本兼容格式（你当前项目现有脚本可直接吃）
    train_unsafe = [x["prompt"] for x in train_samples if x["label"] == 0]
    train_safe = [x["prompt"] for x in train_samples if x["label"] == 1]
    test_unsafe = [x["prompt"] for x in test_samples if x["label"] == 0]
    test_safe = [x["prompt"] for x in test_samples if x["label"] == 1]

    write_txt(out_dir / "text_compat/data/custom.txt", train_unsafe)
    write_txt(out_dir / "text_compat/data_harmless/custom.txt", train_safe)
    write_txt(out_dir / "text_compat/data/testset.txt", test_unsafe)
    write_txt(out_dir / "text_compat/data_harmless/testset.txt", test_safe)

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump({
            "total": len(samples),
            "train": len(train_samples),
            "test": len(test_samples),
            "train_unsafe": len(train_unsafe),
            "train_safe": len(train_safe),
            "test_unsafe": len(test_unsafe),
            "test_safe": len(test_safe),
        }, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Output:", out_dir.resolve())


if __name__ == "__main__":
    main()
