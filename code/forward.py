import os
import json
import pandas as pd
import numpy as np
import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoImageProcessor, LlavaForConditionalGeneration, LlavaProcessor
import torch
import logging
from tqdm import tqdm
import warnings
from utils import patch_open, logging_cuda_memory_usage
from utils import DEFAULT_SYSTEM_PROMPT, SHORT_SYSTEM_PROMPT, MISTRAL_SYSTEM_PROMPT
from utils import infer_mm_dataset_name, load_mm_rows
from safetensors.torch import save_file
import gc
import random
from matplotlib import pyplot as plt


logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")


def prepend_sys_prompt(sentence, args):
    messages = [{'role': 'user', 'content': sentence.strip()}]
    messages_with_default = [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}] + messages
    messages_with_short = [{'role': 'system', 'content': SHORT_SYSTEM_PROMPT}] + messages
    messages_with_mistral = [{'role': 'system', 'content': MISTRAL_SYSTEM_PROMPT}] + messages
    return messages, messages_with_default, messages_with_short, messages_with_mistral


def build_llava_prompt(messages):
    system_text = "\n".join([m["content"] for m in messages if m["role"] == "system"]).strip()
    user_text = "\n".join([m["content"] for m in messages if m["role"] == "user"]).strip()
    merged_user = f"{system_text}\n\n{user_text}".strip() if len(system_text) > 0 else user_text
    return f"USER: {merged_user}\nASSISTANT:"


def build_llava_text_inputs_for_messages(tokenizer, messages, device):
    input_text = build_llava_prompt(messages)
    tokenized = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
    )
    return tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)


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
    return image_features.to(device=llm_device, dtype=model.dtype)


def forward(model, toker, messages, is_llava=False, enable_vision=False, llava_processor=None, image_path=None):
    if is_llava:
        llm_device = model.get_input_embeddings().weight.device
        input_ids, text_attention_mask = build_llava_text_inputs_for_messages(toker, messages, llm_device)
        if enable_vision:
            if image_path is None or llava_processor is None:
                raise ValueError("LLaVA vision forward requires image_path and llava_processor")
            visual_embeds = encode_llava_visual_tokens(model, llava_processor, [image_path])
            text_embeds = model.get_input_embeddings()(input_ids)
            prefix_mask = torch.ones(
                visual_embeds.size(0),
                visual_embeds.size(1),
                dtype=text_attention_mask.dtype,
                device=llm_device,
            )
            outputs = model(
                inputs_embeds=torch.cat([visual_embeds, text_embeds], dim=1),
                attention_mask=torch.cat([prefix_mask, text_attention_mask], dim=1),
                return_dict=True,
                output_hidden_states=True,
            )
        else:
            outputs = model(
                input_ids,
                attention_mask=text_attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
    else:
        input_text = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_device = model.device
        input_ids = torch.tensor(
            toker.convert_tokens_to_ids(toker.tokenize(input_text)),
            dtype=torch.long,
        ).unsqueeze(0).to(model_device)

        outputs = model(
            input_ids,
            attention_mask=input_ids.new_ones(input_ids.size(), dtype=torch.long),
            return_dict=True,
            output_hidden_states=True,
        )
    # We only keep the final token representation from the last decoder layer,
    # which is the only hidden state consumed by estimate.py.
    return outputs.hidden_states[-1][0, -1:].detach().half().cpu()


def main():
    patch_open()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--use_harmless", action="store_true")
    parser.add_argument("--enable_vision", action="store_true")
    parser.add_argument("--mm_jsonl", type=str, default=None)
    parser.add_argument("--output_path", type=str, default='./hidden_states')
    args = parser.parse_args()

    if args.mm_jsonl is None or not os.path.exists(args.mm_jsonl):
        raise ValueError("--mm_jsonl must point to an existing multimodal jsonl file")
    if not args.enable_vision:
        raise ValueError("--mm_jsonl in forward.py requires --enable_vision so hidden states come from image+text")

    # prepare model
    model_name = args.model_name = args.pretrained_model_path.split('/')[-1]
    is_llava = 'llava' in model_name.lower()

    if is_llava:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.pretrained_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            use_safetensors=True,
            device_map="auto",
        )
        try:
            processor = AutoProcessor.from_pretrained(args.pretrained_model_path, use_fast=False)
            llava_processor = processor
        except Exception:
            llava_image_processor = AutoImageProcessor.from_pretrained(args.pretrained_model_path)
            llava_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast=False)
            llava_processor = LlavaProcessor(image_processor=llava_image_processor, tokenizer=llava_tokenizer)
        toker = llava_processor.tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            use_safetensors=True,
            device_map="auto",
            attn_implementation="eager"
        )
        toker = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast='Orca-2-' not in model_name)
        llava_processor = None

    logging.info(f"Model name: {model_name}")
    logging.info(f"Model size: {model.get_memory_footprint()/1e9}")
    logging_cuda_memory_usage()

    config = model.config
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None and hasattr(config, "text_config"):
        num_layers = getattr(config.text_config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Cannot infer num_hidden_layers from model config")

    if not is_llava:
        if 'Llama-2-' in model_name and '-chat' in model_name:
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
        generation_config = json.load(open(generation_config_file))
        chat_template_file = generation_config['chat_template']
        chat_template = open(chat_template_file).read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        toker.chat_template = chat_template

    # prepare data
    all_image_paths = None
    dataset = infer_mm_dataset_name(args.mm_jsonl)
    mm_rows = load_mm_rows(
        args.mm_jsonl,
        label_filter=1 if args.use_harmless else 0,
        require_label=True,
        require_existing_images=True,
    )
    all_queries = [row["question"] for row in mm_rows]
    all_image_paths = [row["image_path"] for row in mm_rows]
    if args.use_harmless:
        args.output_path += "_harmless"
    os.makedirs(f"{args.output_path}", exist_ok=True)

    # prepend sys prompt
    n_queries = len(all_queries)

    all_messages = [prepend_sys_prompt(l, args) for l in all_queries]
    all_messages_with_mistral = [l[3] for l in all_messages]
    all_messages_with_short = [l[2] for l in all_messages]
    all_messages_with_default = [l[1] for l in all_messages]
    all_messages = [l[0] for l in all_messages]
    logging.info(f"Running")
    tensors = {}
    for idx, messages in tqdm(enumerate(all_messages),
                              total=len(all_messages), dynamic_ncols=True):
        image_path = all_image_paths[idx] if all_image_paths is not None else None
        final_hidden_state = forward(
            model,
            toker,
            messages,
            is_llava=is_llava,
            enable_vision=args.enable_vision,
            llava_processor=llava_processor,
            image_path=image_path,
        )
        tensors[f'sample.{idx}_layer.{num_layers-1}'] = final_hidden_state
    save_file(tensors, f'{args.output_path}/{model_name}_{dataset}.safetensors')

    tensors = {}
    for idx, messages_with_default in tqdm(enumerate(all_messages_with_default),
                                       total=len(all_messages_with_default), dynamic_ncols=True):
        image_path = all_image_paths[idx] if all_image_paths is not None else None
        final_hidden_state = forward(
            model,
            toker,
            messages_with_default,
            is_llava=is_llava,
            enable_vision=args.enable_vision,
            llava_processor=llava_processor,
            image_path=image_path,
        )
        tensors[f'sample.{idx}_layer.{num_layers-1}'] = final_hidden_state
    save_file(tensors, f'{args.output_path}/{model_name}_with_default_{dataset}.safetensors')

    tensors = {}
    for idx, messages_with_short in tqdm(enumerate(all_messages_with_short),
                                       total=len(all_messages_with_short), dynamic_ncols=True):
        image_path = all_image_paths[idx] if all_image_paths is not None else None
        final_hidden_state = forward(
            model,
            toker,
            messages_with_short,
            is_llava=is_llava,
            enable_vision=args.enable_vision,
            llava_processor=llava_processor,
            image_path=image_path,
        )
        tensors[f'sample.{idx}_layer.{num_layers-1}'] = final_hidden_state
    save_file(tensors, f'{args.output_path}/{model_name}_with_short_{dataset}.safetensors')

    tensors = {}
    for idx, messages_with_mistral in tqdm(enumerate(all_messages_with_mistral),
                                       total=len(all_messages_with_mistral), dynamic_ncols=True):
        image_path = all_image_paths[idx] if all_image_paths is not None else None
        final_hidden_state = forward(
            model,
            toker,
            messages_with_mistral,
            is_llava=is_llava,
            enable_vision=args.enable_vision,
            llava_processor=llava_processor,
            image_path=image_path,
        )
        tensors[f'sample.{idx}_layer.{num_layers-1}'] = final_hidden_state
    save_file(tensors, f'{args.output_path}/{model_name}_with_mistral_{dataset}.safetensors')

    logging_cuda_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
