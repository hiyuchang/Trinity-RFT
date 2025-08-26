from typing import Any, Dict, Optional

from verl.utils import hf_processor
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.dataset.vision_utils import preprocess, process_image, process_video, process_minicpmo_data, init_minicpmo_config


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