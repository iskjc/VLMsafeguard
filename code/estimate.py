import os
import json
import csv
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoConfig
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from tqdm import tqdm
from scipy.stats import ttest_1samp
import warnings
from utils import patch_open, logging_cuda_memory_usage, get_following_indices
from utils import PCA_DIM
from utils import infer_mm_dataset_name, load_mm_rows
from safetensors import safe_open
import gc
import random
from matplotlib import pyplot as plt
from safetensors.torch import save_file
from sklearn.decomposition import PCA
from copy import deepcopy
from utils import gram_schmidt


logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")


def kmeans_smoothing(coordinates, values, k=5):
    max_distance = torch.max(torch.norm(coordinates.unsqueeze(0) - coordinates.unsqueeze(1), dim=-1))
    smoothed_values = []
    for i in range(len(coordinates)):
        if k == 1:
            smoothed_value = values[i]
        else:
            current_coord = coordinates[i]
            distances = torch.norm(coordinates - current_coord, dim=1)
            _, indices = torch.topk(distances, k=k, largest=False)
            weights = torch.exp(-distances[indices] / max_distance / 0.2)
            smoothed_value = torch.sum(values[indices] * weights) / torch.sum(weights)
        smoothed_values.append(smoothed_value.item())
    smoothed_values = np.array(smoothed_values)
    return smoothed_values


MAX_EPOCHES = 10000


def main():
    patch_open()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--config", type=str, choices=["greedy", "sampling"], required=True)
    parser.add_argument("--output_path", type=str, default='./estimations')
    parser.add_argument("--system_prompt_type", type=str, choices=['all'], required=True)
    parser.add_argument("--mm_jsonl", type=str, default='./data/data_vlguard/processed/train_mm.jsonl')
    parser.add_argument("--n_splits", type=int, default=10)
    args = parser.parse_args()

    # logging args
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # prepare data
    if args.mm_jsonl is None or not os.path.exists(args.mm_jsonl):
        raise ValueError("--mm_jsonl must point to an existing multimodal jsonl file for estimate.py")

    dataset = infer_mm_dataset_name(args.mm_jsonl)
    harmful_rows = load_mm_rows(args.mm_jsonl, label_filter=0, require_label=True, require_existing_images=True)
    harmless_rows = load_mm_rows(args.mm_jsonl, label_filter=1, require_label=True, require_existing_images=True)
    all_queries = [row["question"] for row in harmful_rows]
    all_queries_harmless = [row["question"] for row in harmless_rows]
    logging.info(f"Loaded multimodal train data from {args.mm_jsonl}")
    logging.info(f"Resolved estimation dataset tag: {dataset}")

    n_queries = len(all_queries)
    n_queries_harmless = len(all_queries_harmless)
    logging.info(f"Harmful queries: {n_queries}, harmless queries: {n_queries_harmless}")

    logging_cuda_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()

    logging.info(args.pretrained_model_path)

    # prepare model
    model_name = args.pretrained_model_path.split('/')[-1]
    config = AutoConfig.from_pretrained(args.pretrained_model_path)
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None and hasattr(config, "text_config"):
        num_layers = getattr(config.text_config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Cannot infer num_hidden_layers from model config")
    os.makedirs(f'{args.output_path}/{model_name}_{args.system_prompt_type}', exist_ok=True)

    # harmful
    logging.info(f"Running harmful")
    hidden_states = safe_open(f'hidden_states/{model_name}_{dataset}.safetensors',
                                framework='pt', device=0)
    hidden_states_with_default = safe_open(f'hidden_states/{model_name}_with_default_{dataset}.safetensors',
                                        framework='pt', device=0)
    hidden_states_with_short = safe_open(f'hidden_states/{model_name}_with_short_{dataset}.safetensors',
                                        framework='pt', device=0)
    hidden_states_with_mistral = safe_open(f'hidden_states/{model_name}_with_mistral_{dataset}.safetensors',
                                        framework='pt', device=0)
    all_hidden_states = []
    all_hidden_states_with_default = []
    all_hidden_states_with_short = []
    all_hidden_states_with_mistral = []
    for idx, query in enumerate(all_queries):
        tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
        tmp_hidden_states_with_default = hidden_states_with_default.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
        tmp_hidden_states_with_short = hidden_states_with_short.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
        tmp_hidden_states_with_mistral = hidden_states_with_mistral.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
        all_hidden_states.append(tmp_hidden_states)
        all_hidden_states_with_default.append(tmp_hidden_states_with_default)
        all_hidden_states_with_short.append(tmp_hidden_states_with_short)
        all_hidden_states_with_mistral.append(tmp_hidden_states_with_mistral)

    # harmless
    logging.info(f"Running harmless")
    hidden_states = safe_open(f'hidden_states_harmless/{model_name}_{dataset}.safetensors',
                                framework='pt', device=0)
    hidden_states_with_default = safe_open(f'hidden_states_harmless/{model_name}_with_default_{dataset}.safetensors',
                                        framework='pt', device=0)
    hidden_states_with_short = safe_open(f'hidden_states_harmless/{model_name}_with_short_{dataset}.safetensors',
                                        framework='pt', device=0)
    hidden_states_with_mistral = safe_open(f'hidden_states_harmless/{model_name}_with_mistral_{dataset}.safetensors',
                                        framework='pt', device=0)
    all_hidden_states_harmless = []
    all_hidden_states_with_default_harmless = []
    all_hidden_states_with_short_harmless = []
    all_hidden_states_with_mistral_harmless = []
    for idx, query_harmless in enumerate(all_queries_harmless):
        tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
        tmp_hidden_states_with_default = hidden_states_with_default.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
        tmp_hidden_states_with_short = hidden_states_with_short.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
        tmp_hidden_states_with_mistral = hidden_states_with_mistral.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
        all_hidden_states_harmless.append(tmp_hidden_states)
        all_hidden_states_with_default_harmless.append(tmp_hidden_states_with_default)
        all_hidden_states_with_short_harmless.append(tmp_hidden_states_with_short)
        all_hidden_states_with_mistral_harmless.append(tmp_hidden_states_with_mistral)


    all_hidden_states = torch.stack(all_hidden_states)
    all_hidden_states_with_default = torch.stack(all_hidden_states_with_default)
    all_hidden_states_with_short = torch.stack(all_hidden_states_with_short)
    all_hidden_states_with_mistral = torch.stack(all_hidden_states_with_mistral)
    all_hidden_states_harmless = torch.stack(all_hidden_states_harmless)
    all_hidden_states_with_default_harmless = torch.stack(all_hidden_states_with_default_harmless)
    all_hidden_states_with_short_harmless = torch.stack(all_hidden_states_with_short_harmless)
    all_hidden_states_with_mistral_harmless = torch.stack(all_hidden_states_with_mistral_harmless)

    scores = get_following_indices(
        model_name, dataset=dataset, config=args.config, use_harmless=False, return_only_scores=True)
    scores_harmless = get_following_indices(
        model_name, dataset=dataset, config=args.config, use_harmless=True, return_only_scores=True)
    scores_with_default = get_following_indices(
        model_name, dataset=dataset, config=args.config, use_default_prompt=True, use_harmless=False, return_only_scores=True)
    scores_with_default_harmless = get_following_indices(
        model_name, dataset=dataset, config=args.config, use_default_prompt=True, use_harmless=True, return_only_scores=True)
    scores_with_short = get_following_indices(
        model_name, dataset=dataset, config=args.config, use_short_prompt=True, use_harmless=False, return_only_scores=True)
    scores_with_short_harmless = get_following_indices(
        model_name, dataset=dataset, config=args.config, use_short_prompt=True, use_harmless=True, return_only_scores=True)
    scores_with_mistral = get_following_indices(
        model_name, dataset=dataset, config=args.config, use_mistral_prompt=True, use_harmless=False, return_only_scores=True)
    scores_with_mistral_harmless = get_following_indices(
        model_name, dataset=dataset, config=args.config, use_mistral_prompt=True, use_harmless=True, return_only_scores=True)


    scores = torch.tensor(scores, device='cuda', dtype=torch.float)
    scores_harmless = torch.tensor(scores_harmless, device='cuda', dtype=torch.float)
    scores_with_default = torch.tensor(scores_with_default, device='cuda', dtype=torch.float)
    scores_with_default_harmless = torch.tensor(scores_with_default_harmless, device='cuda', dtype=torch.float)
    scores_with_short = torch.tensor(scores_with_short, device='cuda', dtype=torch.float)
    scores_with_short_harmless = torch.tensor(scores_with_short_harmless, device='cuda', dtype=torch.float)
    scores_with_mistral = torch.tensor(scores_with_mistral, device='cuda', dtype=torch.float)
    scores_with_mistral_harmless = torch.tensor(scores_with_mistral_harmless, device='cuda', dtype=torch.float)

    hidden_state_groups = [
        all_hidden_states,
        all_hidden_states_with_default,
        all_hidden_states_with_short,
        all_hidden_states_with_mistral,
        all_hidden_states_harmless,
        all_hidden_states_with_default_harmless,
        all_hidden_states_with_short_harmless,
        all_hidden_states_with_mistral_harmless,
    ]
    score_groups = [
        scores,
        scores_with_default,
        scores_with_short,
        scores_with_mistral,
        scores_harmless,
        scores_with_default_harmless,
        scores_with_short_harmless,
        scores_with_mistral_harmless,
    ]
    harmful_group_count = sum(group.size(0) for group in hidden_state_groups[:4])
    harmless_group_count = sum(group.size(0) for group in hidden_state_groups[4:])
    total_sample_count = harmful_group_count + harmless_group_count

    for group_name, states, score_tensor in zip([
            "harmful/base",
            "harmful/default",
            "harmful/short",
            "harmful/mistral",
            "harmless/base",
            "harmless/default",
            "harmless/short",
            "harmless/mistral",
        ], hidden_state_groups, score_groups):
        if states.size(0) != score_tensor.size(0):
            raise ValueError(
                f"Mismatched sample count for {group_name}: "
                f"{states.size(0)} hidden states vs {score_tensor.size(0)} scores")


    hidden_states = torch.cat([
        all_hidden_states_harmless,
        all_hidden_states_with_default_harmless,
        all_hidden_states_with_short_harmless,
        all_hidden_states_with_mistral_harmless,
        all_hidden_states,
        all_hidden_states_with_default,
        all_hidden_states_with_short,
        all_hidden_states_with_mistral,
    ], dim=0).float()
    pca = PCA(PCA_DIM, random_state=42)
    pca.fit(hidden_states.cpu().numpy())
    logging.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}, sum: {np.sum(pca.explained_variance_ratio_)}")

    mean = torch.tensor(pca.mean_, dtype=torch.float, device='cuda')
    V = torch.tensor(pca.components_.T, dtype=torch.float, device='cuda')
    n = V.size(0)
    basis = [V[:, i] for i in range(V.size(1))]
    set_seed(42)
    all_vectors = torch.randn((n*2, n), device='cuda', dtype=torch.double)
    orthogonal_basis = gram_schmidt(all_vectors, basis, n)
    save_file({'mean': mean, 'V': orthogonal_basis}, f'{args.output_path}/{model_name}_{args.system_prompt_type}/transform.safetensors')


    def train_model(model, train_X, train_Y, test_X=None, test_Y=None):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        min_epoch = 0
        min_test_bce_loss = 1e5
        min_test_mse_loss_by_bce_loss = 1e5
        epoch = 0
        test_loss_drop_times = 0
        best_model_copy = deepcopy(model)
        while True:
            model.train()
            optimizer.zero_grad()
            train_logits = model(train_X).squeeze(-1)
            bce_loss = F.binary_cross_entropy_with_logits(train_logits, train_Y)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            bce_loss.backward()
            optimizer.step()
            epoch += 1

            if test_X is not None:
                with torch.no_grad():
                    model.eval()
                    test_logits = model(test_X).squeeze(-1)
                    test_bce_loss = F.binary_cross_entropy_with_logits(test_logits, test_Y).detach()
                    test_mse_loss = F.mse_loss(test_logits.sigmoid(), test_Y).detach()
            else:
                test_bce_loss = bce_loss.detach()
                test_mse_loss = F.mse_loss(train_logits.sigmoid(), train_Y).detach()
            if min_test_bce_loss is None or test_bce_loss < min_test_bce_loss - 1e-4:
                min_test_bce_loss = test_bce_loss
                min_test_mse_loss_by_bce_loss = test_mse_loss
                best_model_copy = deepcopy(model)
                test_loss_drop_times = 0
                min_epoch = epoch
            else:
                test_loss_drop_times += 1

            if epoch == MAX_EPOCHES or test_loss_drop_times == 3:
                logging.info(f"Final epoch {min_epoch}: bce_loss {min_test_bce_loss.item()}, mse_loss {min_test_mse_loss_by_bce_loss.item()}")
                break
        return best_model_copy


    total_indices = list(range(total_sample_count))
    set_seed(42)
    random.shuffle(total_indices)
    split_indices = [indices.tolist() for indices in np.array_split(total_indices, args.n_splits)]

    train_hidden_states = torch.cat(hidden_state_groups, dim=0).float()
    refusal_targets = torch.cat(score_groups, dim=0)
    harmfulness_targets = torch.cat([
        torch.zeros(harmful_group_count, device='cuda', dtype=torch.float),
        torch.ones(harmless_group_count, device='cuda', dtype=torch.float),
    ], dim=0)

    logging.info(f"Training refusal model")
    refusal_linear_weights = []
    refusal_linear_biases = []
    for split_idx in range(args.n_splits):
        if args.n_splits > 1:
            test_indices = split_indices[split_idx]
            train_indices = []
            for other_split_idx, split in enumerate(split_indices):
                if other_split_idx != split_idx:
                    train_indices.extend(split)
        else:
            test_indices = None
            train_indices = total_indices

        train_X = train_hidden_states[train_indices]
        train_X = torch.matmul(train_X - mean, V)

        train_Y = refusal_targets[train_indices]
        #train_Y = torch.tensor(kmeans_smoothing(train_X, train_Y), device=train_X.device, dtype=torch.float)

        if test_indices is not None:
            test_X = train_hidden_states[test_indices]
            test_X = torch.matmul(test_X - mean, V)

            test_Y = refusal_targets[test_indices]
        else:
            test_X = None
            test_Y = None

        model_refusal = nn.Linear(PCA_DIM, 1).to('cuda')
        model_refusal = train_model(model_refusal, train_X, train_Y, test_X, test_Y)
        refusal_linear_weights.append(model_refusal.weight.detach())
        refusal_linear_biases.append(model_refusal.bias.detach())


    logging.info(f"Training harmfulness model")
    harmfulness_linear_weights = []
    harmfulness_linear_biases = []
    for split_idx in range(args.n_splits):
        if args.n_splits > 1:
            test_indices = split_indices[split_idx]
            train_indices = []
            for other_split_idx, split in enumerate(split_indices):
                if other_split_idx != split_idx:
                    train_indices.extend(split)
        else:
            test_indices = None
            train_indices = total_indices

        train_X = train_hidden_states[train_indices]
        train_X = torch.matmul(train_X - mean, V)

        train_Y = harmfulness_targets[train_indices]

        if test_indices is not None:
            test_X = train_hidden_states[test_indices]
            test_X = torch.matmul(test_X - mean, V)

            test_Y = harmfulness_targets[test_indices]
        else:
            test_X = None
            test_Y = None

        model_harmfulness = nn.Linear(PCA_DIM, 1).to('cuda')
        model_harmfulness = train_model(model_harmfulness, train_X, train_Y, test_X, test_Y)
        harmfulness_linear_weights.append(model_harmfulness.weight.detach())
        harmfulness_linear_biases.append(model_harmfulness.bias.detach())

    refusal_linear_weight = torch.stack(refusal_linear_weights).cpu()
    refusal_linear_bias = torch.stack(refusal_linear_biases).cpu()
    harmfulness_linear_weight = torch.stack(harmfulness_linear_weights).cpu()
    harmfulness_linear_bias = torch.stack(harmfulness_linear_biases).cpu()

    save_file({'weight': refusal_linear_weight, 'bias': refusal_linear_bias},
              f'{args.output_path}/{model_name}_{args.system_prompt_type}/refusal.safetensors')
    save_file({'weight': harmfulness_linear_weight, 'bias': harmfulness_linear_bias},
              f'{args.output_path}/{model_name}_{args.system_prompt_type}/harmfulness.safetensors')

    logging.info(f"Training finished")

    logging_cuda_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
