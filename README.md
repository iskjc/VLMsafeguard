# SAFE VLM
---

### HOW TO RUN
```python
cd code

python build_data/prepare_vlguard.py \
  --raw_dir /home/srj/2safevlm/2vlmsafe/data/vlguard/raw \
  --image_root /home/srj/2safevlm/2vlmsafe/data/vlguard/images \
  --out_dir /home/srj/VLMsafeguard/code/data/data_vlguard/processed \
  --train_ratio 0.8 \
  --seed 42
 
```
```python
python convert_gqa_to_mm_jsonl.py   --input /s/datasets/gqa/llava_gqa_testdev_balanced.jsonl   --output /home/srj/VLMsafeguard/code/data/gqa_build/llava_gqa_testdev_bal
anced_mm.jsonl   --answers-json /s/datasets/gqa/testdev_balanced_predictions.json
```



```python
python generate.py --pretrained_model_path $MODEL --use_sampling --n_samples 25
python generate.py --pretrained_model_path $MODEL --use_sampling --n_samples 25 --use_default_prompt
python generate.py --pretrained_model_path $MODEL --use_sampling --n_samples 25 --use_short_prompt
python generate.py --pretrained_model_path $MODEL --use_sampling --n_samples 25 --use_mistral_prompt
```
```python
python evaluate.py --config sampling --evaluator_path $EVAL \
  --model_names \
  llava-1.5-7b-hf \
  llava-1.5-7b-hf_with_default \
  llava-1.5-7b-hf_with_short \
  llava-1.5-7b-hf_with_mistral
```
```python
python generate.py --pretrained_model_path $MODEL --use_sampling --n_samples 25 --use_harmless
python generate.py --pretrained_model_path $MODEL --use_sampling --n_samples 25 --use_harmless --use_default_prompt
python generate.py --pretrained_model_path $MODEL --use_sampling --n_samples 25 --use_harmless --use_short_prompt
python generate.py --pretrained_model_path $MODEL --use_sampling --n_samples 25 --use_harmless --use_mistral_prompt
```
```python
python evaluate.py --config sampling --use_harmless \
  --model_names \
  llava-1.5-7b-hf \
  llava-1.5-7b-hf_with_default \
  llava-1.5-7b-hf_with_short \
  llava-1.5-7b-hf_with_mistral
```
```python
python estimate.py --pretrained_model_path $MODEL --config sampling --system_prompt_type all
```
---
## train soft prompt
python train.py \
  --pretrained_model_path "$MODEL" \
  --config sampling \
  --system_prompt_type all \
  --prompt_length default \
  --enable_vision \
  --mm_jsonl /home/srj/VLMsafeguard/code/data/data_vlguard/processed/train_mm.jsonl \
  --batch_size 2 \
  --num_epochs 10 \
  --output_path ./trained_prompts_mm
---

# vlguard

## hard
python generate.py \
  --pretrained_model_path "$MODEL" \
  --enable_vision \
  --mm_jsonl /home/srj/VLMsafeguard/code/data/data_vlguard/processed/test_mm.jsonl \
  --use_default_prompt \
  --n_samples 1 \
  --max_new_tokens 100 \
  --output_path ./outputs_vlguard_testmm_v2
---
## soft
python generate.py \
  --pretrained_model_path "$MODEL" \
  --enable_vision \
  --mm_jsonl /home/srj/VLMsafeguard/code/data/data_vlguard/processed/test_mm.jsonl \
  --use_soft_prompt \
  --system_prompt_type all \
  --prompt_length default \
  --soft_prompt_path /home/srj/VLMsafeguard/code/trained_prompts_mm/llava-1.5-7b-hf/type.all_length.default.safetensors \
  --n_samples 1 \
  --max_new_tokens 100 \
  --output_path ./outputs_vlguard_testmm_soft_v1

---
python compare_vlguard_outputs.py \
  --mm_jsonl /home/srj/VLMsafeguard/code/data/data_vlguard/processed/test_mm.jsonl \
  --baseline_csv /home/srj/VLMsafeguard/code/outputs_vlguard_testmm_v2/llava-1.5-7b-hf_with_default_vlguard_test/output_greedy.csv \
  --soft_csv /home/srj/VLMsafeguard/code/outputs_vlguard_testmm_soft_v1/llava-1.5-7b-hf_with_soft_all_default_vlguard_test/output_greedy.csv \
  --out_json /home/srj/VLMsafeguard/code/eval_results/vlguard_compare_default_vs_soft.json
---


# gqa
```python
python generate.py \
  --pretrained_model_path "$MODEL" \
  --enable_vision \
  --mm_jsonl /home/srj/VLMsafeguard/code/data/gqa_build/llava_gqa_testdev_balanced_mm.jsonl \
  --use_default_prompt \
  --n_samples 1 \
  --output_path ./outputs_gqa


```

# mme
```python
 python generate.py   --pretrained_model_path "$MODEL"   --enable_vision   --mm_jsonl /home/srj/VLMsafeguard/code/data/mme_build/llava_mme_mm.jsonl   --use_default_prompt   --n_samples 1   --output_path ./outputs_mme
 ```

```python
python generate.py \
  --pretrained_model_path "$MODEL" \
  --enable_vision \
  --mm_jsonl "/home/srj/VLMsafeguard/code/data/mme_build/llava_mme_mm.jsonl" \
  --use_soft_prompt \
  --system_prompt_type all \
  --prompt_length default \
  --soft_prompt_path "/home/srj/VLMsafeguard/code/trained_prompts_mm/llava-1.5-7b-hf/type.all_length.default.safetensors" \
  --n_samples 1 \
  --max_new_tokens 100 \
  --output_path "/home/srj/VLMsafeguard/code/outputs_mme_soft"
  ```
