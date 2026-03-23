# SAFE VLM
---

### HOW TO RUN
```python
cd /home/srj/VLMsafeguard/code
export MODEL="/s/models/llava-series-llava-1.5-7b-hf"
export EVAL="/s/models/LLamaGuard-7b"
export MM_JSONL="/home/srj/VLMsafeguard/code/data/data_vlguard/processed/train_mm.jsonl"
export MODEL_NAME="$(basename "$MODEL")"
```
```python
python3 generate.py --pretrained_model_path "$MODEL" --enable_vision --mm_jsonl "$MM_JSONL" --use_sampling --n_samples 25
python3 generate.py --pretrained_model_path "$MODEL" --enable_vision --mm_jsonl "$MM_JSONL" --use_sampling --n_samples 25 --use_default_prompt
python3 generate.py --pretrained_model_path "$MODEL" --enable_vision --mm_jsonl "$MM_JSONL" --use_sampling --n_samples 25 --use_short_prompt
python3 generate.py --pretrained_model_path "$MODEL" --enable_vision --mm_jsonl "$MM_JSONL" --use_sampling --n_samples 25 --use_mistral_prompt
```
```python
# Step 2: 评估 refusal/follow 分数（写入 eval_results/ 和 eval_results_harmless/）
python3 evaluate.py --config sampling --evaluator_path "$EVAL" --mm_jsonl "$MM_JSONL" \
  --model_names "$MODEL_NAME" "${MODEL_NAME}_with_default" "${MODEL_NAME}_with_short" "${MODEL_NAME}_with_mistral"

python3 evaluate.py --config sampling --use_harmless --mm_jsonl "$MM_JSONL" \
  --model_names "$MODEL_NAME" "${MODEL_NAME}_with_default" "${MODEL_NAME}_with_short" "${MODEL_NAME}_with_mistral"
```
```python
# Step 3: 提取 hidden states（写入 hidden_states/ 和 hidden_states_harmless/）
python3 forward.py --pretrained_model_path "$MODEL" --enable_vision --mm_jsonl "$MM_JSONL"
python3 forward.py --pretrained_model_path "$MODEL" --enable_vision --mm_jsonl "$MM_JSONL" --use_harmless
```
```python
# Step 4: PCA 估计（读取 eval_results + hidden_states）
python3 estimate.py --pretrained_model_path "$MODEL" --config sampling --system_prompt_type all --mm_jsonl "$MM_JSONL"
```

# Step 5: 训练 soft prompt
```python
bash VLMsafeguard/code/scripts/multimodal/train_soft_prompt_mm_sweep.sh
```