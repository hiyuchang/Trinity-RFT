from typing import Any, Dict, Optional
import torch

from verl.utils import hf_processor
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.dataset.vision_utils import preprocess, process_image, process_video, process_minicpmo_data, init_minicpmo_config

def _simple_compute_position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    与 HuggingFace 常见实现一致：基于 attention_mask 计算增量 position_ids。
    返回形状：(1, seq_len)，用于非 Qwen2-VL 等常规情形。
    """
    # attention_mask: (1, seq_len)
    # 把 pad 位置置 0，非 pad 位置是 1，然后做 cumsum-1
    pos = (attention_mask.cumsum(dim=1) - 1).clamp_min(0)
    return pos  # (1, seq_len)

def _left_pad_and_truncate(input_ids: torch.Tensor,
                           attention_mask: torch.Tensor,
                           max_length: int,
                           pad_token_id: int,
                           left_pad: bool = True,
                           truncation: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    仿 verl_F.postprocess_data 的最小化实现：左侧 padding + 需要时截断。
    仅用于与 position_ids 的长度对齐；不会返回给调用方。
    """
    bsz, seqlen = input_ids.shape
    if truncation and seqlen > max_length:
        # 左截断：保留右侧
        input_ids = input_ids[:, -max_length:]
        attention_mask = attention_mask[:, -max_length:]
        seqlen = max_length
    if seqlen < max_length:
        pad_len = max_length - seqlen
        pad_ids = torch.full((bsz, pad_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        pad_mask = torch.zeros((bsz, pad_len), dtype=attention_mask.dtype, device=attention_mask.device)
        if left_pad:
            input_ids = torch.cat([pad_ids, input_ids], dim=1)
            attention_mask = torch.cat([pad_mask, attention_mask], dim=1)
        else:
            input_ids = torch.cat([input_ids, pad_ids], dim=1)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
    return input_ids, attention_mask


def build_multi_modal_inputs(
    prompt: str,
    tokenizer: Any,
    raw_mm_data: Optional[Dict[str, Any]],
    config: Dict[str, Any],
    logger: Any,
    **kwargs,
) -> Dict[str, Any]:
    """
    Args:
        prompt: input text which is already applied with chat template
        TODO
    """
    local_path = copy_local_path_from_hdfs(config.model.path)
    processor = hf_processor(local_path, use_fast=True)

    if processor.__class__.__name__ == "MiniCPMVImageProcessor":
        if kwargs.get("minicpmo_config") is None:
            logger.warning("minicpmo_config is not provided, using default config")

        minicpmo_config = init_minicpmo_config(processor, kwargs.get("minicpmo_config"))
        
        model_inputs, multi_modal_data, prompt = process_minicpmo_data(
            mm_data=raw_mm_data, 
            messages=prompt, # TODO: check this
            tokenizer=tokenizer, 
            minicpmo_config=minicpmo_config, 
            max_prompt_length=config.get("max_prompt_length"),
            truncation=True, 
            logger=logger
        )
    else:
        raw_images, raw_videos = raw_mm_data["image"], raw_mm_data["video"]
        
        multi_modal_data = {}
        images = [process_image(image) for image in raw_images]
        multi_modal_data["image"] = images

        videos = [process_video(video) for video in raw_videos]
        multi_modal_data["video"] = [video.numpy() for video in videos]

        model_inputs = processor(text=[prompt], images=images, videos=videos, return_tensors="pt")

    input_ids = model_inputs.pop("input_ids")
    attention_mask = model_inputs.pop("attention_mask")

    if "second_per_grid_ts" in model_inputs:
        model_inputs.pop("second_per_grid_ts")

    # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
    multi_modal_inputs = dict(model_inputs)
    # second_per_grid_ts isn't used for training, just for mrope
    multi_modal_inputs.pop("second_per_grid_ts", None)

    return {"multi_modal_inputs": multi_modal_inputs, "multi_modal_data": multi_modal_data, "prompt": prompt, "input_ids": input_ids, "attention_mask": attention_mask}


def process_minicpmo_data(raw_mm_data, messages, tokenizer, minicpmo_config, max_prompt_length, truncation, logger):
    """
    Process data for MiniCPM-o model
    Adapted from verl/utils/dataset/vision_utils.py
    """
    multi_modal_data = {}
    image = process_image(raw_mm_data)
    multi_modal_data["image"] = [image]
    images_dict = {"<image>": image}

    model_inputs = preprocess(
        images_dict,
        messages,
        tokenizer,
        minicpmo_config["transform"],
        query_nums=minicpmo_config["query_nums"],
        slice_config=minicpmo_config["slice_config"],
        llm_type=minicpmo_config["llm_type"],
        patch_size=minicpmo_config["patch_size"],
        batch_vision=minicpmo_config["batch_vision"],
        max_length=max_prompt_length,
        truncation=truncation,
        logger=logger,
    )

    raw_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    raw_prompt = raw_prompt.replace("<image>", "(<image>./</image>)")

    return model_inputs, multi_modal_data, raw_prompt