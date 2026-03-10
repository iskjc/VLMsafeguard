# README

You can find the scripts of running LLMs with human-crafted safety prompts and training continuous safety prompts in `scripts`. Note that for local running you should set the env variable `HF_MODELS` that indicates the save folder of LLMs.


## How to Run Code

To get generation and evaluation results with **human-crafted safety prompts**, run:

```sh
bash scripts/run_mistral-v1.sh
bash scripts/run_mistral-v1_harmless.sh
```

To train **continuous safety prompts**, and then get generation and evaluation results, run:

```sh
bash scripts/forward.sh
bash scripts/forward_harmless.sh
bash scripts/train_mistral-v1.sh
```

You may uncomment the *unlikelihood* line to reproduce the *vanilla Prompt Tuning* baseline.

To **visualize the hidden states with estimated boundaries**, run:

```sh
bash scripts/compare_gather.sh
```


## Acknowledgement

Our code base builds upon the follow repository: https://github.com/Princeton-SysML/Jailbreak_LLM
