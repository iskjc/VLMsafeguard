from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from PIL import Image
Tensor = torch.Tensor


@dataclass
class MMInputs:
    """Container for multimodal inputs passed to LLM."""

    inputs_embeds: Tensor
    attention_mask: Tensor
    text_input_ids: Tensor
    n_visual_tokens: int


class VisionProjector(nn.Module):
    """Project visual hidden states to LLM embedding space."""

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int, projector_type: str = "linear"):
        super().__init__()
        if projector_type == "linear":
            self.net = nn.Linear(vision_hidden_size, llm_hidden_size, bias=False)
        elif projector_type == "mlp2x_gelu":
            self.net = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size, bias=False),
            )
        else:
            raise ValueError(f"Unsupported projector_type: {projector_type}")

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class VisionLanguageAdapter(nn.Module):
    """
    Vision encoder + projector wrapper.

    Expected flow:
    1) image/pixel_values -> vision_model
    2) last_hidden_state -> projector
    3) [B, N_vis, D_llm] ready to concatenate with text embeddings
    """

    def __init__(self, vision_model: nn.Module, projector: VisionProjector):
        super().__init__()
        self.vision_model = vision_model
        self.projector = projector

    def encode_visual_tokens(self, pixel_values: Tensor) -> Tensor:
        outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
        if hasattr(outputs, "last_hidden_state"):
            vis = outputs.last_hidden_state
        elif hasattr(outputs, "vision_model_output") and hasattr(outputs.vision_model_output, "last_hidden_state"):
            vis = outputs.vision_model_output.last_hidden_state
        else:
            raise ValueError("Cannot find vision last_hidden_state in vision model outputs")


    @torch.no_grad()
    def preprocess_images(self, image_paths: Sequence[str], image_processor) -> Tensor:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = image_processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]


def build_mm_inputs(
    llm_model: nn.Module,
    text_input_ids: Tensor,
    visual_embeds: Tensor,
) -> MMInputs:
    """
    Build unified model inputs for LLM:
    [visual_tokens] + [prompt_tokens + text_tokens]

    Args:
        llm_model: AutoModelForCausalLM instance.
        text_input_ids: [B, T_text] token ids from chat template.
        visual_embeds: [B, T_vis, D] projected visual tokens.
    """

    if text_input_ids.dim() != 2:
        raise ValueError("text_input_ids must be [B, T_text]")
    if visual_embeds.dim() != 3:
        raise ValueError("visual_embeds must be [B, T_vis, D]")

    input_emb = llm_model.get_input_embeddings()(text_input_ids)
    if input_emb.size(0) != visual_embeds.size(0):
        raise ValueError("Batch size mismatch between text_input_ids and visual_embeds")
    if input_emb.size(-1) != visual_embeds.size(-1):
        raise ValueError("Hidden size mismatch between text embedding and visual embedding")

    inputs_embeds = torch.cat([visual_embeds, input_emb], dim=1)
    attention_mask = torch.ones(
        inputs_embeds.size(0),
        inputs_embeds.size(1),
        dtype=torch.long,
        device=inputs_embeds.device,
    )
    return MMInputs(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        text_input_ids=text_input_ids,
        n_visual_tokens=visual_embeds.size(1),
    )


def prepare_text_input_ids(
    tokenizer,
    messages,
    device: Union[str, torch.device],
) -> Tensor:
    """
    Convert chat-template messages to token ids.
    Keeps behavior close to existing generate.py implementation.
    """

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_text)),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    return input_ids


def load_vision_components(
    vision_model_path: str,
    llm_hidden_size: int,
    projector_type: str = "linear",
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[nn.Module, object, VisionProjector]:
    """
    Lazy loader to avoid forcing vision dependencies when not used.
    Returns (vision_model, image_processor, projector).
    """

    from transformers import AutoModel, AutoImageProcessor

    vision_model = AutoModel.from_pretrained(vision_model_path)
    image_processor = AutoImageProcessor.from_pretrained(vision_model_path)

    vision_hidden_size = getattr(vision_model.config, "hidden_size", None)
    if vision_hidden_size is None and hasattr(vision_model.config, "vision_config"):
        vision_hidden_size = getattr(vision_model.config.vision_config, "hidden_size", None)
    if vision_hidden_size is None:
        raise ValueError("Cannot infer vision hidden size from vision model config")

    projector = VisionProjector(
        vision_hidden_size=vision_hidden_size,
        llm_hidden_size=llm_hidden_size,
        projector_type=projector_type,
    )

    if device is not None:
        vision_model = vision_model.to(device)
        projector = projector.to(device)
    if dtype is not None:
        vision_model = vision_model.to(dtype=dtype)
        projector = projector.to(dtype=dtype)
    return vision_model, image_processor, projector
