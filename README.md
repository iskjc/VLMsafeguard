# SAFE VLM
---

### HOW TO RUN
```python
cd code

MODEL=/s/models/llava-series/llava-1.5-7b-hf
EVAL=/s/models/LlamaGuard-7b   
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