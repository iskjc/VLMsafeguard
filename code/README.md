# Code Layout

Multimodal entry points now live in [`scripts/multimodal`](/home/srj/VLMsafeguard/code/scripts/multimodal).

Key directories:

- [`outputs`](/home/srj/VLMsafeguard/code/outputs): generation results grouped by dataset
- [`eval_results`](/home/srj/VLMsafeguard/code/eval_results): evaluation artifacts grouped by dataset
- [`scripts/multimodal`](/home/srj/VLMsafeguard/code/scripts/multimodal): active multimodal train / generate / eval scripts

Common workflows:

- Train soft prompt:
  `bash code/scripts/multimodal/train_soft_prompt_mm.sh`
- Run MME:
  `bash code/scripts/multimodal/generate_mme_hard.sh`
  `bash code/scripts/multimodal/generate_mme_soft.sh`
- Export + score MME:
  `python3 code/scripts/multimodal/export_mme_for_eval_tool.py ...`
  `bash code/scripts/multimodal/eval_mme_results.sh code/eval_results/mme/soft_for_eval_tool /s/datasets/MME/eval_tool`
- Run VLGuard test:
  `bash code/scripts/multimodal/generate_vlguard_test_hard.sh`
  `bash code/scripts/multimodal/generate_vlguard_test_soft.sh`
  `bash code/scripts/multimodal/eval_vlguard_compare.sh`
- Run GQA:
  `bash code/scripts/multimodal/generate_gqa_hard.sh`
  `bash code/scripts/multimodal/eval_gqa_answer_match.sh`

COCO2014 wrappers are included, but they expect you to provide a prepared `mm_jsonl` file, for example by overriding `MM_JSONL`.
