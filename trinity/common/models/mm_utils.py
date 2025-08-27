from typing import Any, Dict, Optional

from verl.utils import hf_processor
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.dataset.vision_utils import (
    process_image,
    process_video,
)


def build_multi_modal_inputs(
    prompt: str,
    raw_mm_data: Optional[Dict[str, Any]],
    config: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """
    Build multi-modal inputs for model
    Adapted from: verl/utils/dataset/rl_dataset.py
    """
    local_path = copy_local_path_from_hdfs(config.model_path)
    processor = hf_processor(local_path, use_fast=True)
    # processor_type = processor.image_processor.__class__.__name__ if processor is not None else None

    if prompt is None:
        raise ValueError("Prompt is required for build multi-modal inputs")

    raw_images, raw_videos = None, None
    if "image" in raw_mm_data:
        raw_images = raw_mm_data["image"]
    if "video" in raw_mm_data:
        raw_videos = raw_mm_data["video"]

    multi_modal_data = {}
    images, videos = None, None
    if raw_images is not None:
        images = [process_image(image) for image in raw_images]
        multi_modal_data["image"] = images
    if raw_videos is not None:
        videos = [process_video(video) for video in raw_videos]
        multi_modal_data["video"] = [video.numpy() for video in videos]

    model_inputs = processor(text=[prompt], images=images, videos=videos, return_tensors="pt")

    input_ids = model_inputs.pop("input_ids", None)  # TODO: check
    attention_mask = model_inputs.pop("attention_mask", None)

    if "second_per_grid_ts" in model_inputs:
        model_inputs.pop("second_per_grid_ts")

    # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
    multi_modal_inputs = dict(model_inputs)

    # TODO: check
    if (
        processor is not None
        and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__
    ):
        from verl.models.transformers.qwen2_vl import get_rope_index

        position_ids = [
            get_rope_index(
                processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
        ]  # (1, 3, seq_len)

    # from trinity.utils.log import get_logger  # For debug
    # logger = get_logger(__name__)
    # logger.debug(f"!!!!!!! multi_modal_inputs: {multi_modal_inputs.keys()}")
    # # logger.debug(f"!!!!!!! multi_modal_data: {multi_modal_data}")
    # logger.debug(f"!!!!!!! prompt: {prompt}")

    return {
        "prompt": prompt,
        "multi_modal_inputs": multi_modal_inputs,
        "multi_modal_data": multi_modal_data,
    }


def attach_images_to_messages(messages, raw_mm_data):
    # 深拷贝一份，避免原数据被污染
    new_msgs = [dict(m) for m in messages]
    imgs = (raw_mm_data or {}).get("image") or []
    if not imgs:
        return new_msgs

    # 找到最后一条 user 消息，把图片贴上去（按需你也可选择首条或所有 user）
    for i in range(len(new_msgs) - 1, -1, -1):
        if new_msgs[i].get("role") == "user":
            content = new_msgs[i].get("content", "")
            items = []
            if isinstance(content, str):
                # 去掉用户文本里“手写”的占位符，避免和模型占位符冲突/重复
                text = content.replace("<image>", "").replace("<|image_pad|>", "").strip()
                if text:
                    items.append({"type": "text", "text": text})
            elif isinstance(content, list):
                # 把 list 里可能的纯字符串转成 text item，已有 dict 则原样保留
                for c in content:
                    if isinstance(c, str):
                        t = c.replace("<image>", "").replace("<|image_pad|>", "").strip()
                        if t:
                            items.append({"type": "text", "text": t})
                    elif isinstance(c, dict):
                        items.append(c)

            # 追加所有图片（多图时依次追加）
            for img in imgs:
                items.append({"type": "image", "image": img})

            new_msgs[i]["content"] = items
            break

    return new_msgs
