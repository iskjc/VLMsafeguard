import os
import json
import pandas as pd
import numpy as np
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoImageProcessor,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    set_seed,
)
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
import torch.nn as nn
from typing import Union
import logging
from tqdm import tqdm
import warnings
from utils import patch_open, logging_cuda_memory_usage
from utils import DEFAULT_SYSTEM_PROMPT, SHORT_SYSTEM_PROMPT, MISTRAL_SYSTEM_PROMPT
import gc
import random
from multiprocessing.pool import ThreadPool
from safetensors import safe_open
from functools import partial
from mm_adapter import (
    load_vision_components,
    VisionLanguageAdapter,
    prepare_text_input_ids,
    build_mm_inputs,
)

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")

BAD_WORDS = [
    '\nHello', '\nHi', # for Orca-2
    ' ON', "I' " # for vicuna
]

def load_checkpoint_state_dict(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    return ckpt["projector"] if isinstance(ckpt, dict) and "projector" in ckpt else ckpt


def prepend_sys_prompt(sentence, args):
    messages = [{'role': 'user', 'content': sentence.strip()}]
    if args.use_soft_prompt:
        messages = [{'role': 'system', 'content': ''.join([f'<soft_prompt_{i}>' for i in range(args.soft_prompt.size(0))])}] + messages
    elif args.use_default_prompt:
        messages = [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}] + messages
    elif args.use_short_prompt:
        messages = [{'role': 'system', 'content': SHORT_SYSTEM_PROMPT}] + messages
    elif args.use_mistral_prompt:
        messages = [{'role': 'system', 'content': MISTRAL_SYSTEM_PROMPT}] + messages
    return messages


def process_soft_prompt_as_word_embedding(
    model: PreTrainedModel,
    toker: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    soft_prompt: torch.nn.Parameter
) -> nn.Module:
    # We embed soft prompt into input word embedding and safe it
    # When loaded later, simply call model.set_input_embeddings()
    config = model.config
    padding_idx = config.pad_token_id

    old_toker_size = len(toker)
    toker.add_tokens([f'<soft_prompt_{i}>' for i in range(soft_prompt.size(0))], special_tokens=True)
    new_toker_size = len(toker)

    old_input_embeddings = model.get_input_embeddings()
    embedding_dim = old_input_embeddings.embedding_dim
    old_num_embeddings = old_input_embeddings.num_embeddings
    new_num_embeddings = max(new_toker_size, old_num_embeddings)

    new_input_embeddings = nn.Embedding(new_num_embeddings, embedding_dim, padding_idx)
    new_input_embeddings.weight.data[:old_toker_size] = old_input_embeddings.weight.data[:old_toker_size]
    new_input_embeddings.weight.data[old_toker_size:new_toker_size] = soft_prompt.data.to('cpu')
    return toker, new_input_embeddings


def generate(inputs, model, toker, max_new_tokens, n_samples, temp, top_p, stop_token_ids, stop_str,
             enable_vision=False,vl_adapter=None,image_processor=None,image_path=None,
             is_llava=False,llava_processor=None):
    qdx, payload = inputs
    if len(payload) == 4:
        seed, query, messages, sample_image_path = payload
    else:
        seed, query, messages = payload
        sample_image_path = None

    if seed is None:
        set_seed(qdx)
    else:
        set_seed(seed)

    if is_llava:
        # Keep this prompt format simple and robust for llava-1.5 checkpoints.
        system_text = "\n".join([m["content"] for m in messages if m["role"] == "system"]).strip()
        user_text = "\n".join([m["content"] for m in messages if m["role"] == "user"]).strip()
        merged_user = f"{system_text}\n\n{user_text}".strip() if len(system_text) > 0 else user_text
        if enable_vision:
            input_text = f"USER: <image>\n{merged_user}\nASSISTANT:"
        else:
            input_text = f"USER: {merged_user}\nASSISTANT:"
    else:
        input_text = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # directly tokenizing words would produce an extra space, so remove it
    bad_words_ids = []
    if not is_llava:
        for t in BAD_WORDS:
            ids = toker.convert_tokens_to_ids(toker.tokenize(t)[1:])
            if len(ids) > 0:
                bad_words_ids.append(ids)
    if len(bad_words_ids) == 0:
        bad_words_ids = None

    if is_llava:
        llava_device = next(model.parameters()).device
        if enable_vision:
            actual_image_path = sample_image_path if sample_image_path is not None else image_path
            if actual_image_path is None:
                raise ValueError("LLaVA with --enable_vision requires --image_path or --mm_jsonl with image_path")
            from PIL import Image
            pil_image = Image.open(actual_image_path).convert("RGB")
            model_inputs = llava_processor(images=pil_image, text=input_text, return_tensors="pt")
        else:
            model_inputs = llava_processor(text=input_text, return_tensors="pt")

        safe_model_inputs = {}
        for k, v in model_inputs.items():
            if v is None:
                continue
            if not torch.is_tensor(v):
                safe_model_inputs[k] = v
                continue
            if torch.is_floating_point(v):
                safe_model_inputs[k] = v.to(device=llava_device, dtype=model.dtype)
            else:
                safe_model_inputs[k] = v.to(device=llava_device)
        model_inputs = safe_model_inputs
        prompt_len = model_inputs["input_ids"].size(1)
        generations = model.generate(
            **model_inputs,
            min_new_tokens=10,
            max_new_tokens=max_new_tokens,
            do_sample=True if temp > 0 else False,
            temperature=temp if temp > 0 else 1.0,
            top_p=top_p,
            top_k=50,
            num_return_sequences=n_samples,
            eos_token_id=stop_token_ids,
            pad_token_id=toker.eos_token_id,
            return_dict_in_generate=True,
        )
    elif enable_vision:
        actual_image_path = sample_image_path if sample_image_path is not None else image_path
        if actual_image_path is None:
            raise ValueError("--enable_vision=True requires --image_path (or --mm_jsonl rows with image_path)")
        text_input_ids = prepare_text_input_ids(tokenizer=toker, messages=messages, device=model.device)
        pixel_values=vl_adapter.preprocess_images([actual_image_path],image_processor)
        visual_embeds=vl_adapter.encode_visual_tokens(pixel_values)
        mm_inputs=build_mm_inputs(
            llm_model=model,
            text_input_ids=text_input_ids,
            visual_embeds=visual_embeds,
        )

        generations = model.generate(
            inputs_embeds=mm_inputs.inputs_embeds,
            attention_mask=mm_inputs.attention_mask,
            min_new_tokens=10,
            max_new_tokens=max_new_tokens,
            do_sample=True if temp > 0 else False,
            temperature=temp if temp > 0 else 1.0,
            top_p=top_p,
            top_k=50, 
            num_return_sequences=n_samples,
            eos_token_id=stop_token_ids,
            pad_token_id=toker.eos_token_id,
            return_dict_in_generate=True,
            bad_words_ids=bad_words_ids,
        )
        prompt_len=mm_inputs.inputs_embeds.size(1)
    else:
        input_ids=torch.tensor(
            toker.convert_tokens_to_ids(toker.tokenize(input_text)),
            dtype=torch.long,
        ).unsqueeze(0).to(model.device)

        generations = model.generate(
            input_ids,
            attention_mask=input_ids.new_ones(input_ids.size(), dtype=torch.long),
            min_new_tokens=10,
            max_new_tokens=max_new_tokens,
            do_sample=True if temp > 0 else False,
            temperature=temp if temp > 0 else 1.0,
            top_p=top_p,
            top_k=50,
            num_return_sequences=n_samples,
            eos_token_id=stop_token_ids,
            pad_token_id=toker.eos_token_id,
            return_dict_in_generate=True,
            bad_words_ids=bad_words_ids,
        )   
        prompt_len = input_ids.size(1)


    sequences=generations.sequences
    if sequences.size(1) > prompt_len:
        generations = sequences[..., prompt_len:]
    else:
        generations = sequences

    generations = generations.tolist()
    generated_texts = []
    for generation in generations:
        gen_tokens = []
        for token in generation:
            if token in stop_token_ids or token == toker.eos_token_id:
                break
            gen_tokens.append(token)

        text = toker.decode(gen_tokens, skip_special_tokens=True)
        if stop_str is not None:
            text = text.split(stop_str)[0]
        text = text.strip()

        while (
            not any(text.endswith(e) for e in ['.', '?', '!']) or
            (len(text) > 1 and text[-1] == '.' and text[-2].isdigit())
        ):
            if len(text.split('\n')) > 1:
                text = '\n'.join(text.split('\n')[:-1])
                text = text.strip()
                continue
            break

        generated_texts.append(text)

    return qdx, query, input_text, generated_texts


def main():
    patch_open()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--use_sampling", action="store_true")

    parser.add_argument("--use_soft_prompt", action="store_true")
    parser.add_argument("--prompt_length", type=str, choices=['default', 'mistral', 'short'])
    parser.add_argument("--system_prompt_type", type=str, choices=['all', 'default', 'mistral', 'short'])
    parser.add_argument("--do_data_ablation", action="store_true")
    parser.add_argument("--do_unlikelihood", action="store_true")
    parser.add_argument("--ablate_norm", action="store_true")
    parser.add_argument("--ablate_refu", action="store_true")
    parser.add_argument("--ablate_harm", action="store_true")

    parser.add_argument("--use_default_prompt", action='store_true')
    parser.add_argument("--use_short_prompt", action='store_true')
    parser.add_argument("--use_mistral_prompt", action='store_true')

    parser.add_argument("--use_malicious", action="store_true")
    parser.add_argument("--use_advbench", action="store_true")
    parser.add_argument("--use_alpaca", action="store_true")
    parser.add_argument("--use_gcg", action="store_true")
    parser.add_argument("--use_harmless", action="store_true")
    parser.add_argument("--use_testset", action="store_true")
    parser.add_argument("--enable_vision", action="store_true")
    parser.add_argument("--vision_model_path", type=str, default=None)
    parser.add_argument("--vision_projector_type", type=str, choices=["linear", "mlp2x_gelu"], default="linear")
    parser.add_argument("--projector_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default='./outputs')
    parser.add_argument("--mm_jsonl", type=str, default=None)

    args = parser.parse_args()

    if sum([args.use_soft_prompt,
            args.use_default_prompt,
            args.use_short_prompt, args.use_mistral_prompt]) > 1:
        raise ValueError("Only one of --use_soft_prompt, --use_default_prompt, --use_short/--use_mistral_prompt can be set to True")
    if not args.use_sampling and args.n_samples > 1:
        raise ValueError("n_samples must be 1 in greedy decoding")
    if sum([args.use_malicious, args.use_advbench, args.use_alpaca]) > 1:
        raise ValueError("Only one of --use_malicious/--use_advbench/--use_alpaca can be set to True")
    if any([args.use_malicious, args.use_advbench, args.use_alpaca]) and args.use_harmless:
        raise ValueError("Only one of --use_malicious/--use_advbench/--use_alpaca and --use_harmless can be set to True")
    if any([args.use_malicious, args.use_advbench, args.use_alpaca]) and args.use_testset:
        raise ValueError("Only one of --use_malicious/--use_advbench/--use_alpaca and --use_testset can be set to True")
    if args.use_testset and not args.use_harmless:
        raise ValueError("--use_testset must be used with --use_harmless")
    if args.use_soft_prompt and (args.prompt_length is None or args.system_prompt_type is None):
        raise ValueError("--use_soft_prompt requires both --prompt_length and --system_prompt_type")

    # prepare toker
    model_name = args.model_name = args.pretrained_model_path.split('/')[-1]
    toker = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast=False)
    is_llava = ('llava' in model_name.lower())

    if is_llava:
        generation_config = {"stop_token_ids": None, "stop_str": None}
    elif 'Llama-2-' in model_name and '-chat' in model_name:
        generation_config_file = './generation_configs/llama-2-chat.json'
    elif 'CodeLlama-' in model_name and '-Instruct' in model_name:
        generation_config_file = './generation_configs/llama-2-chat.json'
    elif 'Orca-2-' in model_name:
        generation_config_file = './generation_configs/orca-2.json'
    elif 'Mistral-' in model_name and '-Instruct' in model_name:
        generation_config_file = './generation_configs/mistral-instruct.json'
    elif 'vicuna-' in model_name:
        generation_config_file = './generation_configs/vicuna.json'
    elif 'openchat-' in model_name:
        generation_config_file = './generation_configs/openchat.json'
    else:
        raise ValueError(f"Unsupported or untuned model: {model_name}")
    if not is_llava:
        generation_config = json.load(open(generation_config_file))
        chat_template_file = generation_config['chat_template']
        chat_template = open(chat_template_file).read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        toker.chat_template = chat_template

    stop_token_ids = generation_config['stop_token_ids']
    if stop_token_ids is None:
        stop_token_ids = [toker.eos_token_id]
    stop_str = generation_config['stop_str']

    # prepare data
    fname = model_name
    if args.use_soft_prompt:
        if args.do_unlikelihood:
            fname += f"_with_soft_unlikelihood_{args.prompt_length}"
        else:
            fname += f"_with_soft_{args.system_prompt_type}_{args.prompt_length}"
        if args.do_data_ablation:
            fname += '_ablation'
            fname += '_unlikelihood'
        elif args.ablate_norm:
            fname += "_nonorm"
        elif args.ablate_refu:
            fname += "_norefu"
        elif args.ablate_harm:
            fname += "_noharm"
    elif args.use_default_prompt:
        fname += "_with_default"
    elif args.use_short_prompt:
        fname += "_with_short"
    elif args.use_mistral_prompt:
        fname += "_with_mistral"

    if args.use_harmless:
        data_path = './data_harmless'
        args.output_path += "_harmless"
    elif args.use_alpaca:
        data_path = './data_alpaca'
        args.output_path += "_alpaca"
    else:
        data_path = './data'

    if args.use_advbench:
        fname += "_advbench"
        with open(f"{data_path}/advbench.txt") as f:
            lines = f.readlines()[:100]
    elif args.use_malicious:
        fname += "_malicious"
        with open(f"{data_path}/MaliciousInstruct.txt") as f:
            lines = f.readlines()
    elif args.use_gcg:
        fname += "_gcg"
        with open(f"{data_path}/advbench.txt") as f:
            lines = f.readlines()[:100]
        templates = json.load(open(f'{data_path}/gcg.json'))
        for i in range(len(lines)):
            template = templates[str(i)]['final_suffix']
            lines[i] = lines[i] + template
    elif args.use_testset:
        fname += "_testset"
        with open(f"{data_path}/testset.txt") as f:
            lines = f.readlines()
    elif args.use_alpaca:
        fname += "_alpaca"
        with open(f"{data_path}/alpaca_eval.json") as f:
            lines = [e['instruction'] for e in json.load(f)[:100]]
    else:
        fname += "_custom"
        with open(f"{data_path}/custom.txt") as f:
            lines = f.readlines()
    os.makedirs(f"{args.output_path}/{fname}", exist_ok=True)

    # logging args
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    output_file = f"{args.output_path}/{fname}/output_sampling.csv" if args.use_sampling else f"{args.output_path}/{fname}/output_greedy.csv"
    if os.path.exists(output_file):
        logging.info(f"File {output_file} exists, skipping")
        return

    # prepare model
    model_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported()
        and not ((('Orca-2-' in model_name and args.use_soft_prompt)
                  or ('vicuna-' in model_name and not args.use_soft_prompt)
                  ) and args.use_testset)
        else torch.float32
    )
    if is_llava:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.pretrained_model_path,
            torch_dtype=model_dtype,
            use_safetensors=True,
            device_map="auto",
        )
        try:
            llava_processor = AutoProcessor.from_pretrained(
                args.pretrained_model_path,
                use_fast=False,
            )
        except Exception as e:
            logging.warning(f"AutoProcessor load failed ({type(e).__name__}): {e}")
            logging.warning("Falling back to LlavaProcessor(image_processor + slow tokenizer).")
            llava_image_processor = AutoImageProcessor.from_pretrained(args.pretrained_model_path)
            llava_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast=False)
            llava_processor = LlavaProcessor(
                image_processor=llava_image_processor,
                tokenizer=llava_tokenizer,
            )
        # Keep tokenizer usage consistent for decode/postprocess path.
        if hasattr(llava_processor, "tokenizer") and llava_processor.tokenizer is not None:
            toker = llava_processor.tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_path,
            torch_dtype=model_dtype,
            use_safetensors=True,
            device_map="auto",
            attn_implementation="eager"
        )
        llava_processor = None

    logging.info(f"Model name: {model_name}")
    logging.info(f"Model size: {model.get_memory_footprint()/1e9}")
    logging_cuda_memory_usage()

    if args.use_soft_prompt:
        if is_llava:
            raise ValueError("--use_soft_prompt is not supported with LLaVA models in this script")
        if args.do_data_ablation:
            soft_prompt_file = f'./trained_prompts_ablation/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}.safetensors'
        elif args.do_unlikelihood:
            soft_prompt_file = f'./trained_prompts_unlikelihood/{model_name}/length.{args.prompt_length}.safetensors'
        elif args.ablate_norm:
            soft_prompt_file = f'./trained_prompts/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_nonorm.safetensors'
        elif args.ablate_refu:
            soft_prompt_file = f'./trained_prompts/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_norefu.safetensors'
        elif args.ablate_harm:
            soft_prompt_file = f'./trained_prompts/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_noharm.safetensors'
        else:
            soft_prompt_file = f'./trained_prompts/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}.safetensors'
        with safe_open(soft_prompt_file, framework='pt') as f:
            soft_prompt = f.get_tensor('soft_prompt')
        args.soft_prompt = soft_prompt
        toker, new_input_embeddings = process_soft_prompt_as_word_embedding(model, toker, soft_prompt)
        model.set_input_embeddings(new_input_embeddings.to(device=model.device, dtype=model.dtype))


    #增加视觉模块
    vl_adapter = None
    image_processor = None
    if args.enable_vision and not is_llava:
        if args.vision_model_path is None:
            raise ValueError("--enable_vision requires --vision_model_path")
        if args.image_path is None and args.mm_jsonl is None:
            raise ValueError("--enable_vision requires --image_path or --mm_jsonl")
        llm_hidden_size = model.get_input_embeddings().embedding_dim
        vision_model, image_processor, projector = load_vision_components(
            vision_model_path=args.vision_model_path,
            llm_hidden_size=llm_hidden_size,
            projector_type=args.vision_projector_type,
            device=model.device,
            dtype=model.dtype,
        )
        if args.projector_path is not None:
            state_dict = load_checkpoint_state_dict(args.projector_path)
            projector.load_state_dict(state_dict, strict=True)
        vl_adapter = VisionLanguageAdapter(vision_model=vision_model, projector=projector).to(model.device)
        vl_adapter.eval()

    # prepend sys prompt
    all_image_paths = None
    if args.mm_jsonl is not None:
        mm_rows = []
        skipped_missing_images = 0
        with open(args.mm_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "image_path" not in row or "question" not in row:
                    continue
                question = str(row["question"]).strip()
                if len(question) == 0:
                    continue
                if args.enable_vision and not os.path.exists(row["image_path"]):
                    skipped_missing_images += 1
                    continue
                mm_rows.append({"question": question, "image_path": row["image_path"]})

        if len(mm_rows) == 0:
            raise ValueError(f"No valid rows with question/image_path found in {args.mm_jsonl}")
        if skipped_missing_images > 0:
            logging.warning(f"Skipped {skipped_missing_images} rows with missing image files in {args.mm_jsonl}")

        all_queries = [e["question"] for e in mm_rows]
        all_image_paths = [e["image_path"] for e in mm_rows]
    else:
        all_queries = [l.strip() for l in lines]

    all_messages = [prepend_sys_prompt(l, args) for l in all_queries]

    if args.use_gcg:
        with open(f"{data_path}/advbench.txt") as f:
            lines = f.readlines()[:100]
        all_image_paths = None
        all_queries = [l.strip() for l in lines]
        all_messages = [prepend_sys_prompt(l, args) for l in all_queries]

    logging.info(f"Running")
    prompts = []
    inputs = []
    outputs = []
    model.eval()

    if args.use_harmless:
        max_new_tokens = 200
    elif args.use_alpaca:
        max_new_tokens = 1000
    else:
        max_new_tokens = 300

    if args.use_sampling:
        generate_fn = partial(
            generate, model=model, toker=toker,
            max_new_tokens=max_new_tokens,
            n_samples=args.n_samples if args.use_sampling else 1,
            temp=1 if args.use_sampling else 0,
            top_p=0.9 if args.use_sampling else 0,
            stop_token_ids=stop_token_ids, stop_str=stop_str,
            enable_vision=args.enable_vision,
            vl_adapter=vl_adapter,
            image_processor=image_processor,
            image_path=args.image_path,
            is_llava=is_llava,
            llava_processor=llava_processor,
        )
    else:
        generate_fn = partial(
            generate, model=model, toker=toker,
            max_new_tokens=max_new_tokens,
            n_samples=args.n_samples if args.use_sampling else 1,
            temp=1 if args.use_sampling else 0,
            top_p=0.9 if args.use_sampling else 0,
            stop_token_ids=stop_token_ids, stop_str=stop_str,
            enable_vision=args.enable_vision,
            vl_adapter=vl_adapter,
            image_processor=image_processor,
            image_path=args.image_path,
            is_llava=is_llava,
            llava_processor=llava_processor,
        )

    pool = ThreadPool(1)

    seeds = [None] * len(all_queries) # by default, we use qdx
    pbar = tqdm(total=len(all_queries), dynamic_ncols=True)
    
    if all_image_paths is not None:
        packed = zip(seeds, all_queries, all_messages, all_image_paths)
    else:
        packed = zip(seeds, all_queries, all_messages)
    
    try:
        for res in pool.imap(generate_fn, enumerate(packed), chunksize=1):
            qdx, query, input_text, generated_texts = res
            if qdx < 5:
                logging.info(f"\nQuery: {query}")
                logging.info(f"\nInput: {input_text}")
                logging.info(f"\nOutput: {generated_texts[0]}\n")
            inputs.extend([input_text] * args.n_samples)
            outputs.extend(generated_texts)
            prompts.extend([query] * args.n_samples)
            pbar.update(1)
    finally:
        pool.close()
        pool.join()
        pbar.close()

    results = pd.DataFrame()
    results["prompt"] = prompts
    results["input"] = inputs
    results["output"] = outputs
    if args.use_sampling:
        results.to_csv(f"{args.output_path}/{fname}/output_sampling.csv")
    else:
        results.to_csv(f"{args.output_path}/{fname}/output_greedy.csv")

    logging_cuda_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
