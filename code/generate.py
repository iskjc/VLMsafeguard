import os
import json
import pandas as pd
import numpy as np
import argparse
import glob
from PIL import Image
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
from typing import Union, Optional
import logging
from tqdm import tqdm
import warnings
from utils import patch_open, logging_cuda_memory_usage
from utils import DEFAULT_SYSTEM_PROMPT, SHORT_SYSTEM_PROMPT, MISTRAL_SYSTEM_PROMPT
from utils import infer_mm_dataset_name, load_mm_rows
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


def prepend_sys_prompt(sentence, args, is_llava: bool = False):
    messages = [{'role': 'user', 'content': sentence.strip()}]
    if args.use_soft_prompt and not is_llava:
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


def resolve_versioned_soft_prompt_file(prompt_dir: str, save_stem: str, version: Optional[str] = None) -> Optional[str]:
    default_file = os.path.join(prompt_dir, f"{save_stem}.safetensors")
    if version is not None:
        return os.path.join(prompt_dir, f"{save_stem}__{version}.safetensors")
    if os.path.exists(default_file):
        return default_file

    versioned_pattern = os.path.join(prompt_dir, f"{save_stem}__*.safetensors")
    versioned_files = [path for path in glob.glob(versioned_pattern) if os.path.isfile(path)]
    if len(versioned_files) == 0:
        return None

    versioned_files.sort(key=lambda path: (os.path.getmtime(path), path), reverse=True)
    chosen_file = versioned_files[0]
    logging.info(
        "No exact soft prompt found at %s, using latest versioned file: %s",
        default_file,
        chosen_file,
    )
    return chosen_file


def resolve_soft_prompt_file(args, model_name: str, is_llava: bool) -> str:
    if args.soft_prompt_path is not None:
        return args.soft_prompt_path
    if args.do_data_ablation:
        return f'./trained_prompts_ablation/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}.safetensors'
    if args.do_unlikelihood:
        return f'./trained_prompts_unlikelihood/{model_name}/length.{args.prompt_length}.safetensors'
    if args.ablate_norm:
        return f'./trained_prompts/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_nonorm.safetensors'
    if args.ablate_refu:
        return f'./trained_prompts/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_norefu.safetensors'
    if args.ablate_harm:
        return f'./trained_prompts/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_noharm.safetensors'
    save_stem = f'type.{args.system_prompt_type}_length.{args.prompt_length}'
    if is_llava:
        mm_file = resolve_versioned_soft_prompt_file(
            prompt_dir=f'./trained_prompts_mm/{model_name}',
            save_stem=save_stem,
            version=args.soft_prompt_version,
        )
        if mm_file is not None:
            return mm_file
        if args.soft_prompt_version is not None:
            return f'./trained_prompts_mm/{model_name}/{save_stem}__{args.soft_prompt_version}.safetensors'
        default_file = resolve_versioned_soft_prompt_file(
            prompt_dir=f'./trained_prompts/{model_name}',
            save_stem=save_stem,
        )
        if default_file is not None:
            return default_file
        return f'./trained_prompts/{model_name}/{save_stem}.safetensors'
    default_file = resolve_versioned_soft_prompt_file(
        prompt_dir=f'./trained_prompts/{model_name}',
        save_stem=save_stem,
        version=args.soft_prompt_version,
    )
    if default_file is not None:
        return default_file
    if args.soft_prompt_version is not None:
        return f'./trained_prompts/{model_name}/{save_stem}__{args.soft_prompt_version}.safetensors'
    return f'./trained_prompts/{model_name}/{save_stem}.safetensors'


def build_llava_prompt_text(messages):
    system_text = "\n".join([m["content"] for m in messages if m["role"] == "system"]).strip()
    user_text = "\n".join([m["content"] for m in messages if m["role"] == "user"]).strip()
    merged_user = f"{system_text}\n\n{user_text}".strip() if len(system_text) > 0 else user_text
    return f"USER: {merged_user}\nASSISTANT:"


def build_llava_text_inputs_for_messages(tokenizer, messages, device):
    input_text = build_llava_prompt_text(messages)
    tokenized = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
    )
    return input_text, tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)


def encode_llava_visual_tokens(model: LlavaForConditionalGeneration, llava_processor, image_paths):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    pixel_values = llava_processor.image_processor(images=images, return_tensors="pt")["pixel_values"]
    vision_param = next(model.vision_tower.parameters())
    llm_device = model.get_input_embeddings().weight.device
    pixel_values = pixel_values.to(device=vision_param.device, dtype=vision_param.dtype)
    with torch.no_grad():
        image_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[model.config.vision_feature_layer]
        if model.config.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif model.config.vision_feature_select_strategy != "full":
            raise ValueError(f"Unexpected vision_feature_select_strategy: {model.config.vision_feature_select_strategy}")
        image_features = model.multi_modal_projector(selected_image_feature)
    image_features = image_features.to(device=llm_device, dtype=model.dtype)
    return image_features


def generate(inputs, model, toker, max_new_tokens, n_samples, temp, top_p, stop_token_ids, stop_str,
             enable_vision=False,vl_adapter=None,image_processor=None,image_path=None,
             is_llava=False,llava_processor=None,soft_prompt: Optional[torch.Tensor] = None):
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
        merged_text_prompt = build_llava_prompt_text(messages)
        if enable_vision:
            input_text = merged_text_prompt.replace("USER: ", "USER: <image>\n", 1)
        else:
            input_text = merged_text_prompt
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

    if is_llava and soft_prompt is not None:
        llm_device = model.get_input_embeddings().weight.device
        llava_tokenizer = llava_processor.tokenizer if hasattr(llava_processor, "tokenizer") and llava_processor.tokenizer is not None else toker
        _, text_input_ids, text_attention_mask = build_llava_text_inputs_for_messages(
            tokenizer=llava_tokenizer,
            messages=messages,
            device=llm_device,
        )
        text_embeds = model.get_input_embeddings()(text_input_ids)
        bsz = text_embeds.size(0)
        soft_batch = soft_prompt.to(device=llm_device, dtype=model.dtype).unsqueeze(0).repeat(bsz, 1, 1)

        if enable_vision:
            actual_image_path = sample_image_path if sample_image_path is not None else image_path
            if actual_image_path is None:
                raise ValueError("LLaVA with --enable_vision requires --image_path or --mm_jsonl with image_path")
            visual_embeds = encode_llava_visual_tokens(model, llava_processor, [actual_image_path])
            n_vis = visual_embeds.size(1)
            n_prompt = soft_batch.size(1)
            prefix_mask = torch.ones(bsz, n_vis + n_prompt, dtype=text_attention_mask.dtype, device=llm_device)
            inputs_embeds = torch.cat([visual_embeds, soft_batch, text_embeds], dim=1)
            attention_mask = torch.cat([prefix_mask, text_attention_mask], dim=1)
        else:
            n_prompt = soft_batch.size(1)
            prefix_mask = torch.ones(bsz, n_prompt, dtype=text_attention_mask.dtype, device=llm_device)
            inputs_embeds = torch.cat([soft_batch, text_embeds], dim=1)
            attention_mask = torch.cat([prefix_mask, text_attention_mask], dim=1)

        prompt_len = inputs_embeds.size(1)
        llava_lm = model.language_model
        generations = llava_lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            min_new_tokens=1,
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
    elif is_llava:
        llava_device = next(model.parameters()).device
        if enable_vision:
            actual_image_path = sample_image_path if sample_image_path is not None else image_path
            if actual_image_path is None:
                raise ValueError("LLaVA with --enable_vision requires --image_path or --mm_jsonl with image_path")
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
            min_new_tokens=1,
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
            min_new_tokens=1,
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
            min_new_tokens=1,
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
    parser.add_argument("--seed_base", type=int, default=None)

    parser.add_argument("--use_soft_prompt", action="store_true")
    parser.add_argument("--prompt_length", type=str, choices=['default', 'mistral', 'short'])
    parser.add_argument("--system_prompt_type", type=str, choices=['all', 'default', 'mistral', 'short'])
    parser.add_argument("--do_data_ablation", action="store_true")
    parser.add_argument("--do_unlikelihood", action="store_true")
    parser.add_argument("--ablate_norm", action="store_true")
    parser.add_argument("--ablate_refu", action="store_true")
    parser.add_argument("--ablate_harm", action="store_true")
    parser.add_argument("--soft_prompt_path", type=str, default=None)
    parser.add_argument("--soft_prompt_version", type=str, default=None)

    parser.add_argument("--use_default_prompt", action='store_true')
    parser.add_argument("--use_short_prompt", action='store_true')
    parser.add_argument("--use_mistral_prompt", action='store_true')

    parser.add_argument("--enable_vision", action="store_true")
    parser.add_argument("--vision_model_path", type=str, default=None)
    parser.add_argument("--vision_projector_type", type=str, choices=["linear", "mlp2x_gelu"], default="linear")
    parser.add_argument("--projector_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default='./outputs')
    parser.add_argument("--mm_jsonl", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)

    args = parser.parse_args()

    if sum([args.use_soft_prompt,
            args.use_default_prompt,
            args.use_short_prompt, args.use_mistral_prompt]) > 1:
        raise ValueError("Only one of --use_soft_prompt, --use_default_prompt, --use_short/--use_mistral_prompt can be set to True")
    if not args.use_sampling and args.n_samples > 1:
        raise ValueError("n_samples must be 1 in greedy decoding")
    if args.use_soft_prompt and (args.prompt_length is None or args.system_prompt_type is None):
        raise ValueError("--use_soft_prompt requires both --prompt_length and --system_prompt_type")
    if args.mm_jsonl is None or not os.path.exists(args.mm_jsonl):
        raise ValueError("--mm_jsonl must point to an existing multimodal jsonl file")
    if not args.enable_vision:
        raise ValueError("generate.py now only supports multimodal generation and requires --enable_vision")

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

    dataset = infer_mm_dataset_name(args.mm_jsonl)
    fname += f"_{dataset}"
    os.makedirs(f"{args.output_path}/{fname}", exist_ok=True)

    # logging args
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    output_file = f"{args.output_path}/{fname}/output_sampling.csv" if args.use_sampling else f"{args.output_path}/{fname}/output_greedy.csv"
    if os.path.exists(output_file):
        logging.info(f"File {output_file} exists, skipping")
        return

    # prepare model
    prefer_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and not (('Orca-2-' in model_name and args.use_soft_prompt)
                 or ('vicuna-' in model_name and not args.use_soft_prompt))
    )
    if prefer_bf16:
        model_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32
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
    logging.info(f"Model dtype: {model_dtype}")
    logging.info(f"Model size: {model.get_memory_footprint()/1e9}")
    logging_cuda_memory_usage()

    if args.use_soft_prompt:
        soft_prompt_file = resolve_soft_prompt_file(args, model_name=model_name, is_llava=is_llava)
        if not os.path.exists(soft_prompt_file):
            raise ValueError(f"soft prompt file not found: {soft_prompt_file}")
        with safe_open(soft_prompt_file, framework='pt') as f:
            soft_prompt = f.get_tensor('soft_prompt')
        logging.info(f"Loaded soft prompt: {soft_prompt_file}")
        args.soft_prompt = soft_prompt
        if not is_llava:
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
    mm_rows = load_mm_rows(
        args.mm_jsonl,
        label_filter=None,
        require_label=False,
        require_existing_images=True,
    )
    all_queries = [e["question"] for e in mm_rows]
    all_image_paths = [e["image_path"] for e in mm_rows]

    all_messages = [prepend_sys_prompt(l, args, is_llava=is_llava) for l in all_queries]

    logging.info(f"Running")
    prompts = []
    inputs = []
    outputs = []
    model.eval()

    if args.max_new_tokens is not None:
        if args.max_new_tokens <= 0:
            raise ValueError("--max_new_tokens must be positive")
        max_new_tokens = args.max_new_tokens
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
            soft_prompt=args.soft_prompt if args.use_soft_prompt else None,
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
            soft_prompt=args.soft_prompt if args.use_soft_prompt else None,
        )

    pool = ThreadPool(1)

    if args.seed_base is None:
        seeds = [None] * len(all_queries)  # by default, we use qdx
    else:
        seeds = [args.seed_base + i for i in range(len(all_queries))]
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
