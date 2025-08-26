from typing import Any, Dict, List, Optional

from verl.utils import hf_processor
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.dataset.vision_utils import (
    preprocess,
    process_image,
    process_video,
    init_minicpmo_config,
)


def build_multi_modal_inputs(
    tokenizer: Any,
    raw_mm_data: Optional[Dict[str, Any]],
    config: Dict[str, Any],
    logger: Any,
    prompt: str = None,
    messages: List[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Build multi-modal inputs for model
    Adapted from: verl/utils/dataset/rl_dataset.py
    """
    local_path = copy_local_path_from_hdfs(config.model_path)
    processor = hf_processor(local_path, use_fast=True)
    processor_type = processor.image_processor.__class__.__name__ if processor is not None else None

    if processor_type == "MiniCPMVImageProcessor":
        if kwargs.get("minicpmo_config") is None:
            logger.warning("minicpmo_config is not provided, using default config")

        minicpmo_config = init_minicpmo_config(processor, kwargs.get("minicpmo_config"))

        model_inputs, multi_modal_data, prompt = process_minicpmo_data(
            raw_image=raw_mm_data["image"],
            messages=messages,
            prompt=prompt,
            tokenizer=tokenizer,
            minicpmo_config=minicpmo_config,
            max_prompt_length=config.max_prompt_length,
            truncation=True,
            logger=logger,
        )
    else:
        if prompt is None:
            raise ValueError("Prompt is required for build multi-modal inputs")

        raw_images, raw_videos = raw_mm_data["image"], raw_mm_data["video"]

        multi_modal_data = {}
        images, videos = None, None
        if raw_images is not None:
            images = [process_image(image) for image in raw_images]
            multi_modal_data["image"] = images
        if raw_videos is not None:
            videos = [process_video(video) for video in raw_videos]
            multi_modal_data["video"] = [video.numpy() for video in videos]

        model_inputs = processor(text=[prompt], images=images, videos=videos, return_tensors="pt")

    model_inputs.pop("input_ids", None)  # TODO: check
    model_inputs.pop("attention_mask", None)

    if "second_per_grid_ts" in model_inputs:
        model_inputs.pop("second_per_grid_ts")

    # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
    multi_modal_inputs = dict(model_inputs)

    return {
        "prompt": prompt,
        "multi_modal_inputs": multi_modal_inputs,
        "multi_modal_data": multi_modal_data,
    }


def process_minicpmo_data(
    raw_image, messages, prompt, tokenizer, minicpmo_config, max_prompt_length, truncation, logger
):
    """
    Process data for MiniCPM-o model
    Adapted from verl/utils/dataset/vision_utils.py
    """
    multi_modal_data = {}
    image = process_image(raw_image)
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

    prompt = prompt.replace("<image>", "(<image>./</image>)")

    return model_inputs, multi_modal_data, prompt
