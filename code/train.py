import os
import json
import argparse
from typing import Union, Dict, List
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoImageProcessor,
    LlavaProcessor,
    LlavaForConditionalGeneration,
)
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from tqdm import tqdm
import warnings
from utils import patch_open, logging_cuda_memory_usage
from safetensors import safe_open
import gc
import random
from safetensors.torch import save_file
from utils import DEFAULT_SYSTEM_PROMPT, SHORT_SYSTEM_PROMPT, MISTRAL_SYSTEM_PROMPT
from utils import PCA_DIM
from mm_adapter import (
    load_vision_components,
    VisionLanguageAdapter,
)


logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")

BATCH_SIZE = 50
NUM_EPOCHES = 40


def embed_soft_prompt(
    model: PreTrainedModel,
    toker: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    all_messages: List[List[Dict[str, str]]],
    soft_prompt: torch.Tensor
):
    if soft_prompt.device != model.device:
        raise ValueError("soft_prompt must be on the same device as model")
    if soft_prompt.dtype != model.dtype:
        raise ValueError("soft_prompt must be of the same dtype as model")

    if soft_prompt.dim() != 2:
        raise ValueError("soft_prompt must be a 2D tensor")
    if any(len(messages) != 1 for messages in all_messages):
        raise ValueError("all_messages must be a list of single-message lists")
    n_prompt_tokens = soft_prompt.size(0)

    # As system message appears first, we replace the first n_prompt_tokens eos tokens with soft_prompt
    messages_with_eos_placeholder = [[{'role': 'system', 'content': toker.eos_token * n_prompt_tokens}] + e for e in all_messages]
    input_ids = [toker.apply_chat_template(e, add_generation_prompt=True, tokenize=True) for e in messages_with_eos_placeholder]
    input_lengths = [len(e) for e in input_ids]
    max_input_length = max(input_lengths)
    input_ids = [e + [toker.eos_token_id] * (max_input_length - len(e)) for e in input_ids]

    placeholder_start_index = input_ids[0].index(toker.eos_token_id) # all input_ids have the same placeholder_start_index
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds[:, placeholder_start_index:placeholder_start_index+n_prompt_tokens] = soft_prompt.unsqueeze(0).repeat(inputs_embeds.size(0), 1, 1)
    return inputs_embeds, input_lengths


def load_checkpoint_state_dict(path: str):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    return ckpt["projector"] if isinstance(ckpt, dict) and "projector" in ckpt else ckpt


def load_mm_samples(mm_jsonl_path: str):
    samples = []
    with open(mm_jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "question" not in row or "image_path" not in row or "label" not in row:
                continue
            question = str(row["question"]).strip()
            image_path = str(row["image_path"]).strip()
            if len(question) == 0 or len(image_path) == 0:
                continue
            if not os.path.exists(image_path):
                raise ValueError(f"Missing image_path at line {line_idx}: {image_path}")
            label = int(row["label"])
            if label not in [0, 1]:
                raise ValueError(f"label must be 0/1 at line {line_idx}, got {row['label']}")
            samples.append({
                "question": question,
                "image_path": image_path,
                "label": label,
            })
    if len(samples) == 0:
        raise ValueError(f"No valid rows with question/image_path/label found in {mm_jsonl_path}")
    return samples


def build_text_inputs_for_questions(
    model: PreTrainedModel,
    toker: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    questions: List[str],
):
    tokenized = []
    for q in questions:
        messages = [{'role': 'user', 'content': q.strip()}]
        input_text = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = toker.convert_tokens_to_ids(toker.tokenize(input_text))
        tokenized.append(ids)
    text_lens = [len(e) for e in tokenized]
    max_len = max(text_lens)
    input_ids = [e + [toker.eos_token_id] * (max_len - len(e)) for e in tokenized]
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device)
    text_embeds = model.get_input_embeddings()(input_ids)
    return text_embeds, text_lens


def embed_mm_soft_prompt(
    model: PreTrainedModel,
    toker: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    vl_adapter: VisionLanguageAdapter,
    image_processor,
    batch_samples: List[Dict[str, Union[str, int]]],
    soft_prompt: torch.Tensor,
):
    if soft_prompt.device != model.device:
        raise ValueError("soft_prompt must be on the same device as model")
    if soft_prompt.dtype != model.dtype:
        raise ValueError("soft_prompt must be of the same dtype as model")
    if soft_prompt.dim() != 2:
        raise ValueError("soft_prompt must be a 2D tensor")

    image_paths = [e["image_path"] for e in batch_samples]
    questions = [e["question"] for e in batch_samples]
    with torch.no_grad():
        pixel_values = vl_adapter.preprocess_images(image_paths, image_processor)
        visual_embeds = vl_adapter.encode_visual_tokens(pixel_values)

    text_embeds, text_lens = build_text_inputs_for_questions(model, toker, questions)
    bsz = text_embeds.size(0)
    n_vis = visual_embeds.size(1)
    n_prompt = soft_prompt.size(0)
    soft_batch = soft_prompt.unsqueeze(0).repeat(bsz, 1, 1)
    inputs_embeds = torch.cat([visual_embeds, soft_batch, text_embeds], dim=1)
    attention_mask = torch.ones(
        bsz,
        inputs_embeds.size(1),
        dtype=torch.long,
        device=inputs_embeds.device,
    )
    last_token_indices = [n_vis + n_prompt + l - 1 for l in text_lens]
    return inputs_embeds, attention_mask, last_token_indices


def embed_mm_base_inputs(
    model: PreTrainedModel,
    toker: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    vl_adapter: VisionLanguageAdapter,
    image_processor,
    batch_samples: List[Dict[str, Union[str, int]]],
):
    image_paths = [e["image_path"] for e in batch_samples]
    questions = [e["question"] for e in batch_samples]
    with torch.no_grad():
        pixel_values = vl_adapter.preprocess_images(image_paths, image_processor)
        visual_embeds = vl_adapter.encode_visual_tokens(pixel_values)

    text_embeds, text_lens = build_text_inputs_for_questions(model, toker, questions)
    bsz = text_embeds.size(0)
    n_vis = visual_embeds.size(1)
    inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
    attention_mask = torch.ones(
        bsz,
        inputs_embeds.size(1),
        dtype=torch.long,
        device=inputs_embeds.device,
    )
    last_token_indices = [n_vis + l - 1 for l in text_lens]
    return inputs_embeds, attention_mask, last_token_indices


def build_llava_text_inputs_for_questions(tokenizer, questions: List[str], device: torch.device):
    prompts = [f"USER: {q.strip()}\nASSISTANT:" for q in questions]
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    text_lens = attention_mask.sum(dim=1).tolist()
    return input_ids, attention_mask, text_lens


def encode_llava_visual_tokens(model: LlavaForConditionalGeneration, llava_processor, image_paths: List[str]):
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


def embed_llava_soft_prompt(
    model: LlavaForConditionalGeneration,
    llava_processor,
    batch_samples: List[Dict[str, Union[str, int]]],
    soft_prompt: torch.Tensor,
):
    llm_device = model.get_input_embeddings().weight.device
    image_paths = [e["image_path"] for e in batch_samples]
    questions = [e["question"] for e in batch_samples]
    visual_embeds = encode_llava_visual_tokens(model, llava_processor, image_paths)
    input_ids, text_attention_mask, text_lens = build_llava_text_inputs_for_questions(
        tokenizer=llava_processor.tokenizer,
        questions=questions,
        device=llm_device,
    )
    text_embeds = model.get_input_embeddings()(input_ids)
    bsz = text_embeds.size(0)
    n_vis = visual_embeds.size(1)
    n_prompt = soft_prompt.size(0)
    soft_batch = soft_prompt.to(device=llm_device, dtype=model.dtype).unsqueeze(0).repeat(bsz, 1, 1)
    inputs_embeds = torch.cat([visual_embeds, soft_batch, text_embeds], dim=1)
    prefix_mask = torch.ones(bsz, n_vis + n_prompt, dtype=text_attention_mask.dtype, device=llm_device)
    attention_mask = torch.cat([prefix_mask, text_attention_mask], dim=1)
    last_token_indices = [n_vis + n_prompt + l - 1 for l in text_lens]
    return inputs_embeds, attention_mask, last_token_indices


def embed_llava_base_inputs(
    model: LlavaForConditionalGeneration,
    llava_processor,
    batch_samples: List[Dict[str, Union[str, int]]],
):
    llm_device = model.get_input_embeddings().weight.device
    image_paths = [e["image_path"] for e in batch_samples]
    questions = [e["question"] for e in batch_samples]
    visual_embeds = encode_llava_visual_tokens(model, llava_processor, image_paths)
    input_ids, text_attention_mask, text_lens = build_llava_text_inputs_for_questions(
        tokenizer=llava_processor.tokenizer,
        questions=questions,
        device=llm_device,
    )
    text_embeds = model.get_input_embeddings()(input_ids)
    bsz = text_embeds.size(0)
    n_vis = visual_embeds.size(1)
    inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
    prefix_mask = torch.ones(bsz, n_vis, dtype=text_attention_mask.dtype, device=llm_device)
    attention_mask = torch.cat([prefix_mask, text_attention_mask], dim=1)
    last_token_indices = [n_vis + l - 1 for l in text_lens]
    return inputs_embeds, attention_mask, last_token_indices


def get_shuffled_messages_and_labels(all_messages: List[Dict], labels: torch.Tensor, seed=42):
    rng = random.Random(seed)
    assert len(all_messages) == len(labels)
    indices = list(range(len(all_messages)))
    for epoch_idx in range(NUM_EPOCHES):
        rng.shuffle(indices)
        for start in range(0, len(all_messages), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(all_messages))
            batch_idx = indices[start:end]
            yield epoch_idx, [all_messages[i] for i in batch_idx], labels[batch_idx]


def main():
    patch_open()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--config", type=str, choices=["greedy", "sampling"])
    parser.add_argument("--system_prompt_type", type=str, choices=['all', 'default', 'mistral', 'short'], required=True)
    parser.add_argument("--prompt_length", type=str, choices=['default', 'mistral', 'short'], required=True)
    parser.add_argument("--output_path", type=str, default='./trained_prompts')
    parser.add_argument("--ablate_norm", action='store_true')
    parser.add_argument("--ablate_refu", action='store_true')
    parser.add_argument("--ablate_harm", action='store_true')
    parser.add_argument("--enable_vision", action="store_true")
    parser.add_argument("--vision_model_path", type=str, default=None)
    parser.add_argument("--vision_projector_type", type=str, choices=["linear", "mlp2x_gelu"], default="linear")
    parser.add_argument("--projector_path", type=str, default=None)
    parser.add_argument("--mm_jsonl", type=str, default=None)
    args = parser.parse_args()
    model_name = args.pretrained_model_path.split('/')[-1]
    is_llava = 'llava' in model_name.lower()

    if sum([args.ablate_norm, args.ablate_refu, args.ablate_harm]) >= 2:
        raise ValueError("Only one of --ablate_norm, --ablate_refu, --ablate_harm can be set to True")
    if args.enable_vision and args.mm_jsonl is None:
        raise ValueError("--enable_vision requires --mm_jsonl")
    if args.enable_vision and (not is_llava) and args.vision_model_path is None:
        raise ValueError("--enable_vision requires --vision_model_path")
    if is_llava and not args.enable_vision:
        raise ValueError("LLaVA training in this script requires --enable_vision")

    # logging args
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # prepare model
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if is_llava:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.pretrained_model_path,
            torch_dtype=model_dtype,
            use_safetensors=True,
            device_map="auto",
        )
        try:
            llava_processor = AutoProcessor.from_pretrained(args.pretrained_model_path, use_fast=False)
        except Exception as e:
            logging.warning(f"AutoProcessor load failed ({type(e).__name__}): {e}")
            logging.warning("Falling back to LlavaProcessor(image_processor + slow tokenizer).")
            llava_image_processor = AutoImageProcessor.from_pretrained(args.pretrained_model_path)
            llava_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast=False)
            llava_processor = LlavaProcessor(image_processor=llava_image_processor, tokenizer=llava_tokenizer)
        toker = llava_processor.tokenizer
        if toker is None:
            raise ValueError("Cannot load tokenizer from Llava AutoProcessor")
        if toker.pad_token_id is None:
            toker.pad_token = toker.eos_token
        device = model.get_input_embeddings().weight.device
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_path,
            torch_dtype=model_dtype,
            use_safetensors=True,
            device_map="auto",
            attn_implementation="eager"
        )
        toker = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast='Orca-2-' not in model_name)
        llava_processor = None
        device = model.device
    for param in model.parameters():
        param.requires_grad = False

    logging.info(f"Model name: {model_name}")
    logging.info(f"Model size: {model.get_memory_footprint()/1e9}")
    logging_cuda_memory_usage()

    os.makedirs(f'{args.output_path}/{model_name}', exist_ok=True)

    # prepare LinearTransform
    refusal_model = nn.Linear(PCA_DIM, 1)
    with safe_open(f'./estimations/{model_name}_{args.system_prompt_type}/refusal.safetensors', framework='pt') as f:
        weight = f.get_tensor('weight').mean(dim=0)
        bias = f.get_tensor('bias').mean(dim=0)
    refusal_model.load_state_dict({'weight': weight, 'bias': bias})
    refusal_model.float().to(device)
    for param in refusal_model.parameters():
        param.requires_grad = False

    harmfulness_model = nn.Linear(PCA_DIM, 1)
    with safe_open(f'./estimations/{model_name}_{args.system_prompt_type}/harmfulness.safetensors', framework='pt') as f:
        weight = f.get_tensor('weight').mean(dim=0)
        bias = f.get_tensor('bias').mean(dim=0)
    harmfulness_model.load_state_dict({'weight': weight, 'bias': bias})
    harmfulness_model.float().to(device)
    for param in harmfulness_model.parameters():
        param.requires_grad = False

    with safe_open(f'./estimations/{model_name}_{args.system_prompt_type}/transform.safetensors', framework='pt') as f:
        mean = f.get_tensor('mean').float().to(device)
        V = f.get_tensor('V').float().to(device)

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

    vl_adapter = None
    image_processor = None
    if args.enable_vision and (not is_llava):
        llm_hidden_size = model.get_input_embeddings().embedding_dim
        vision_model, image_processor, projector = load_vision_components(
            vision_model_path=args.vision_model_path,
            llm_hidden_size=llm_hidden_size,
            projector_type=args.vision_projector_type,
            device=model.device,
            dtype=model.dtype,
        )
        if args.projector_path is not None:
            projector_state = load_checkpoint_state_dict(args.projector_path)
            projector.load_state_dict(projector_state, strict=True)
        vl_adapter = VisionLanguageAdapter(vision_model=vision_model, projector=projector).to(model.device)
        vl_adapter.eval()
        for param in vl_adapter.parameters():
            param.requires_grad = False

    # prepare soft prompt
    if args.prompt_length == 'default':
        init_ids = toker(DEFAULT_SYSTEM_PROMPT).input_ids[1:]
    elif args.prompt_length == 'short':
        init_ids = toker(SHORT_SYSTEM_PROMPT).input_ids[1:]
    elif args.prompt_length == 'mistral':
        init_ids = toker(MISTRAL_SYSTEM_PROMPT).input_ids[1:]
    init_embeds = model.get_input_embeddings().weight.data[init_ids].detach()
    soft_prompt = nn.Parameter(init_embeds, requires_grad=True).to(device=device, dtype=model.dtype)

    logging.info(f"Other modules loaded")
    logging_cuda_memory_usage()

    # prepare data
    if args.enable_vision:
        raw_mm_samples = load_mm_samples(args.mm_jsonl)
        all_samples = [{
            "sample_id": f"mm_{idx}",
            "question": e["question"],
            "image_path": e["image_path"],
            "label": int(e["label"]),
        } for idx, e in enumerate(raw_mm_samples)]
    else:
        dataset = 'custom'
        with open(f"./data/{dataset}.txt") as f:
            lines = f.readlines()
        with open(f"./data_harmless/{dataset}.txt") as f:
            lines_harmless = f.readlines()

        all_queries = [e.strip() for e in lines if e.strip()]
        all_queries_harmless = [e.strip() for e in lines_harmless if e.strip()]
        all_samples = []
        for idx, q in enumerate(all_queries):
            all_samples.append({
                "sample_id": f"text_unsafe_{idx}",
                "messages": [{'role': 'user', 'content': q}],
                "label": 0,
            })
        for idx, q in enumerate(all_queries_harmless):
            all_samples.append({
                "sample_id": f"text_safe_{idx}",
                "messages": [{'role': 'user', 'content': q}],
                "label": 1,
            })

    labels = torch.tensor([e["label"] for e in all_samples], dtype=torch.float).to(device)

    base_transformeds = {}
    base_refusal_logits = {}
    base_harmfulness_logits = {}
    for sample in tqdm(all_samples, desc="Preparing base states"):
        sample_id = sample["sample_id"]
        if is_llava and args.enable_vision:
            base_inputs_embeds, base_attention_mask, base_last_indices = embed_llava_base_inputs(
                model=model,
                llava_processor=llava_processor,
                batch_samples=[sample],
            )
            last_hidden_state = model.language_model(
                inputs_embeds=base_inputs_embeds,
                attention_mask=base_attention_mask,
                output_hidden_states=True,
            ).hidden_states[-1][0, base_last_indices[0]].unsqueeze(0)
        elif args.enable_vision:
            base_inputs_embeds, base_attention_mask, base_last_indices = embed_mm_base_inputs(
                model=model,
                toker=toker,
                vl_adapter=vl_adapter,
                image_processor=image_processor,
                batch_samples=[sample],
            )
            last_hidden_state = model(
                inputs_embeds=base_inputs_embeds,
                attention_mask=base_attention_mask,
                output_hidden_states=True,
            ).hidden_states[-1][0, base_last_indices[0]].unsqueeze(0)
        else:
            messages = sample["messages"]
            if args.prompt_length == 'default':
                messages = [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}] + messages
            elif args.prompt_length == 'short':
                messages = [{'role': 'system', 'content': SHORT_SYSTEM_PROMPT}] + messages
            elif args.prompt_length == 'mistral':
                messages = [{'role': 'system', 'content': MISTRAL_SYSTEM_PROMPT}] + messages
            input_ids = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=model.device)
            last_hidden_state = model(input_ids, output_hidden_states=True).hidden_states[-1][:, -1]

        transformed = torch.matmul(last_hidden_state.float() - mean, V)
        refusal_logits = refusal_model(transformed[:, :PCA_DIM]).squeeze(-1)
        harmfulness_logits = harmfulness_model(transformed[:, :PCA_DIM]).squeeze(-1)
        base_transformeds[sample_id] = transformed.detach()
        base_refusal_logits[sample_id] = refusal_logits.detach()
        base_harmfulness_logits[sample_id] = harmfulness_logits.detach()

    step = 0
    optimizer = torch.optim.AdamW([soft_prompt], lr=1e-3)
    seed = 42
    for epoch_idx, batch_samples, batch_labels in get_shuffled_messages_and_labels(all_samples, labels, seed=seed):
        batch_ids = [e["sample_id"] for e in batch_samples]
        batch_base_refusal_logits = torch.concat([base_refusal_logits[e] for e in batch_ids], dim=0)
        batch_base_harmfulness_logits = torch.concat([base_harmfulness_logits[e] for e in batch_ids], dim=0)
        optimizer.zero_grad()

        if is_llava and args.enable_vision:
            inputs_embeds, attention_mask, last_token_indices = embed_llava_soft_prompt(
                model=model,
                llava_processor=llava_processor,
                batch_samples=batch_samples,
                soft_prompt=soft_prompt,
            )
            new_hidden_states = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).hidden_states[-1]
            new_last_hidden_states = new_hidden_states[
                torch.arange(len(last_token_indices), device=new_hidden_states.device),
                torch.tensor(last_token_indices, dtype=torch.long, device=new_hidden_states.device),
            ]
        elif args.enable_vision:
            inputs_embeds, attention_mask, last_token_indices = embed_mm_soft_prompt(
                model=model,
                toker=toker,
                vl_adapter=vl_adapter,
                image_processor=image_processor,
                batch_samples=batch_samples,
                soft_prompt=soft_prompt,
            )
            new_hidden_states = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).hidden_states[-1]
            new_last_hidden_states = new_hidden_states[
                torch.arange(len(last_token_indices), device=new_hidden_states.device),
                torch.tensor(last_token_indices, dtype=torch.long, device=new_hidden_states.device),
            ]
        else:
            batch_messages = [e["messages"] for e in batch_samples]
            inputs_embeds, new_input_lengths = embed_soft_prompt(model, toker, batch_messages, soft_prompt)
            new_hidden_states = model(inputs_embeds=inputs_embeds, output_hidden_states=True).hidden_states[-1]
            new_last_hidden_states = new_hidden_states[
                torch.arange(len(new_input_lengths), device=new_hidden_states.device),
                torch.tensor(new_input_lengths, dtype=torch.long, device=new_hidden_states.device) - 1,
            ]

        base_transformed = torch.concat([base_transformeds[e] for e in batch_ids], dim=0)
        new_transformed = torch.matmul(new_last_hidden_states.float() - mean, V)

        norm_loss = torch.mean((new_transformed[:, PCA_DIM:] - base_transformed[:, PCA_DIM:])**2)
        refusal_logits = refusal_model(new_transformed[:, :PCA_DIM]).squeeze(-1) - batch_base_refusal_logits
        refusal_loss = F.binary_cross_entropy_with_logits(refusal_logits, batch_labels)
        harmfulness_logits = harmfulness_model(new_transformed[:, :PCA_DIM]).squeeze(-1) - batch_base_harmfulness_logits
        harmfulness_loss = F.binary_cross_entropy_with_logits(harmfulness_logits, batch_labels)

        if args.ablate_refu:
            total_loss = harmfulness_loss + norm_loss * 1e-3
        elif args.ablate_harm:
            total_loss = refusal_loss + norm_loss * 1e-3
        elif args.ablate_norm:
            total_loss = refusal_loss + harmfulness_loss * 1e-2
        else:
            total_loss = refusal_loss + harmfulness_loss * 1e-2 + norm_loss * 1e-3

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(soft_prompt, 1.0)
        optimizer.step()
        step += 1

        if step % 10 == 0:
            logging.info(f'Step {step}, refusal_loss {refusal_loss.cpu().item()}, harmfulness_loss {harmfulness_loss.cpu().item()}, norm_loss {norm_loss.cpu().item()}')

    soft_prompt = soft_prompt.detach()
    if args.ablate_norm:
        save_file({'soft_prompt': soft_prompt}, f'{args.output_path}/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_nonorm.safetensors')
    elif args.ablate_refu:
        save_file({'soft_prompt': soft_prompt}, f'{args.output_path}/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_norefu.safetensors')
    elif args.ablate_harm:
        save_file({'soft_prompt': soft_prompt}, f'{args.output_path}/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_noharm.safetensors')
    else:
        save_file({'soft_prompt': soft_prompt}, f'{args.output_path}/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}.safetensors')

    logging.info(f"Training finished")

    logging_cuda_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
