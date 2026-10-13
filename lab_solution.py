"""Student-owned implementation for the Qwen3-VL Day 10 lab.

The instructor-provided files under ``scripts/``, ``tests/`` and ``configs/``
remain untouched. Heavy dependencies are imported only by the functions that
need them so the coordinate and reward checks can run on a CPU-only machine.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CLASSES = ("pothole", "crack", "manhole")
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROMPTS = {
    "norm1000": (
        "Inspect this 640x360 dashcam image for road damage. Detect only pothole, "
        "crack, and manhole. Return only a JSON list; each item must be "
        '{"label":"pothole|crack|manhole","bbox_2d":[x1,y1,x2,y2]}. '
        "Coordinates must be integers on the normalized 0-1000 scale, with "
        "[x1,y1,x2,y2] as top-left and bottom-right. Use [] when no damage is visible."
    ),
    "pixel": (
        "Inspect this 640x360 dashcam image for road damage. Detect only pothole, "
        "crack, and manhole. Return only a JSON list; each item must be "
        '{"label":"pothole|crack|manhole","bbox_2d":[x1,y1,x2,y2]}. '
        "Coordinates are absolute image pixels in a 640x360 image, with "
        "[x1,y1,x2,y2] as top-left and bottom-right. Use [] when no damage is visible."
    ),
}


def _coord_limits(coord: str, width: int, height: int) -> tuple[float, float]:
    if coord == "norm1000":
        return 1000.0, 1000.0
    if coord == "pixel":
        return float(width), float(height)
    raise ValueError(f"unsupported coordinate system: {coord!r}")


def yolo_to_box(
    label: str,
    yolo: Sequence[float],
    *,
    width: int = 640,
    height: int = 360,
    coord: str = "norm1000",
) -> dict[str, Any]:
    """Convert one normalized YOLO (xc, yc, w, h) box to xyxy integers."""
    if label not in CLASSES:
        raise ValueError(f"unknown damage class: {label!r}")
    if len(yolo) != 4:
        raise ValueError("YOLO box must contain xc, yc, width, height")
    values = [float(v) for v in yolo]
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
        raise ValueError(f"YOLO values must be finite and in [0, 1]: {yolo!r}")
    x_scale, y_scale = _coord_limits(coord, width, height)
    xc, yc, box_w, box_h = values
    coords = [
        round((xc - box_w / 2) * x_scale),
        round((yc - box_h / 2) * y_scale),
        round((xc + box_w / 2) * x_scale),
        round((yc + box_h / 2) * y_scale),
    ]
    coords[0] = min(max(coords[0], 0), int(x_scale))
    coords[2] = min(max(coords[2], 0), int(x_scale))
    coords[1] = min(max(coords[1], 0), int(y_scale))
    coords[3] = min(max(coords[3], 0), int(y_scale))
    return {"label": label, "bbox_2d": coords}


def objects_to_answer(
    objects: Sequence[Mapping[str, Any]],
    *,
    width: int = 640,
    height: int = 360,
    coord: str = "norm1000",
) -> str:
    """Create deterministic compact JSON supervision, ordered top-to-bottom."""
    boxes = [
        yolo_to_box(
            str(obj["label"]), obj["yolo"], width=width, height=height, coord=coord
        )
        for obj in objects
    ]
    boxes.sort(key=lambda box: (box["bbox_2d"][1], box["bbox_2d"][0], box["label"]))
    return json.dumps(boxes, separators=(",", ":"), ensure_ascii=False)


def inspect_splits(data_root: str | Path = "data") -> dict[str, Any]:
    """Print split counts and verify session isolation without touching source tools."""
    from scripts.labcommon import read_jsonl

    root = Path(data_root)
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        splits[split] = read_jsonl(root / "splits" / f"{split}.jsonl")
        class_counts = {label: 0 for label in CLASSES}
        for row in splits[split]:
            for obj in row["objects"]:
                class_counts[obj["label"]] += 1
        print(f"{split:5s}: {len(splits[split]):4d} ảnh, boxes={class_counts}")

    session_owner: dict[str, str] = {}
    leaks: list[tuple[str, str, str]] = []
    for split, rows in splits.items():
        for row in rows:
            session = row["session"]
            previous = session_owner.setdefault(session, split)
            if previous != split:
                leaks.append((session, previous, split))
    if leaks:
        raise AssertionError(f"sessions appear in multiple splits: {leaks[:5]}")
    print(f"session isolation: OK ({len(session_owner)} sessions)")
    return splits


def preview_ground_truth(
    rows: Sequence[Mapping[str, Any]],
    *,
    data_root: str | Path = "data",
    output_dir: str | Path = "report/figures/ground_truth",
    count: int = 6,
    seed: int = 42,
) -> list[Path]:
    """Draw and save a deterministic set of ground-truth previews."""
    from PIL import Image, ImageDraw

    if not rows:
        raise ValueError("no rows available for ground-truth previews")
    count = min(max(1, count), len(rows))
    chosen = sorted(random.Random(seed).sample(list(rows), count), key=lambda row: row["image_id"])
    root, out = Path(data_root), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    colors = {"pothole": "#ff4d4f", "crack": "#ffd43b", "manhole": "#37b24d"}
    written: list[Path] = []
    for index, row in enumerate(chosen, 1):
        image_path = root / row["image_path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for obj in row["objects"]:
            xc, yc, box_w, box_h = obj["yolo"]
            box = (
                round((xc - box_w / 2) * image.width),
                round((yc - box_h / 2) * image.height),
                round((xc + box_w / 2) * image.width),
                round((yc + box_h / 2) * image.height),
            )
            color = colors[obj["label"]]
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0] + 2, max(0, box[1] - 14)), obj["label"], fill=color)
        target = out / f"gt_{index:02d}_{row['image_id']}.png"
        image.save(target)
        written.append(target)
    return written


def build_messages(image: Any, prompt: str) -> list[dict[str, Any]]:
    """Build one Qwen3-VL user conversation from a PIL image and text prompt."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def encode_example(
    processor: Any,
    image: Any,
    prompt: str,
    answer: str,
) -> dict[str, Any]:
    """Encode one image conversation and mask the prompt from causal-LM loss."""
    import torch

    user = build_messages(image, prompt)[0]
    conversation = [user, {"role": "assistant", "content": answer}]
    prompt_enc = processor.apply_chat_template(
        [user],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_enc = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_len = int(prompt_enc["input_ids"].shape[1])
    if not torch.equal(full_enc["input_ids"][:, :prompt_len], prompt_enc["input_ids"]):
        raise ValueError("chat template changed the user prefix between prompt and full example")
    labels = full_enc["input_ids"].clone()
    labels[:, :prompt_len] = -100
    encoded = {key: value for key, value in full_enc.items() if torch.is_tensor(value)}
    encoded["labels"] = labels
    if not torch.any(labels != -100):
        raise ValueError("answer produced no trainable tokens")
    return encoded


class RoadDamageDataset:
    """Lazy image/text encoding so full-resolution pixels do not fill host RAM."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        processor: Any,
        *,
        data_root: str | Path = "data",
        coord: str = "norm1000",
        fraction: float = 1.0,
        seed: int = 42,
    ) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        self.rows = list(rows)
        if fraction < 1.0:
            count = max(1, round(len(self.rows) * fraction))
            indices = sorted(random.Random(seed).sample(range(len(self.rows)), count))
            self.rows = [self.rows[index] for index in indices]
        self.processor = processor
        self.data_root = Path(data_root)
        self.coord = coord
        self.prompt = PROMPTS[coord]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        row = self.rows[index]
        with Image.open(self.data_root / row["image_path"]) as image:
            image = image.convert("RGB")
            answer = objects_to_answer(
                row["objects"], width=image.width, height=image.height, coord=self.coord
            )
            encoded = encode_example(self.processor, image, self.prompt, answer)
        encoded["image_id"] = row["image_id"]
        return encoded


_LLM_TARGET = re.compile(
    r"(?:^|.*\.)language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
_VISION_TARGET = re.compile(
    r"(?:^|.*\.)visual\.blocks\.\d+\."
    r"(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))$"
)


def find_lora_target_names(module_names: Iterable[str], setup: str) -> list[str]:
    """Return exact Qwen3-VL linear module names for E1/E2/E3 LoRA."""
    setup = setup.upper()
    if setup not in {"E1", "E2", "E3"}:
        raise ValueError("LoRA setup must be E1, E2, or E3")
    patterns = (_LLM_TARGET, _VISION_TARGET) if setup == "E3" else (_LLM_TARGET,)
    return sorted({name for name in module_names if any(p.fullmatch(name) for p in patterns)})


def collate_examples(
    batch: Sequence[Mapping[str, Any]], *, pad_token_id: int
) -> dict[str, Any]:
    """Right-pad text fields; concatenate visual patch tensors without padding."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    if not batch:
        raise ValueError("cannot collate an empty batch")

    def sequence(item: Mapping[str, Any], key: str):
        value = item[key]
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise ValueError(f"{key} must be a 1-D sequence per example, got {tuple(value.shape)}")
        return value

    result = {
        "input_ids": pad_sequence(
            [sequence(item, "input_ids") for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "attention_mask": pad_sequence(
            [sequence(item, "attention_mask") for item in batch],
            batch_first=True,
            padding_value=0,
        ),
        "labels": pad_sequence(
            [sequence(item, "labels") for item in batch],
            batch_first=True,
            padding_value=-100,
        ),
        "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch], dim=0),
    }
    if any("mm_token_type_ids" in item for item in batch):
        if not all("mm_token_type_ids" in item for item in batch):
            raise ValueError("mm_token_type_ids must be present for every Qwen3-VL example")
        result["mm_token_type_ids"] = pad_sequence(
            [sequence(item, "mm_token_type_ids") for item in batch],
            batch_first=True,
            padding_value=0,
        )
    return result


def _completion_text(completion: Any) -> str:
    """Normalize the completion shapes used by TRL's reward callback."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        return str(completion.get("content", ""))
    if isinstance(completion, Sequence) and completion:
        last = completion[-1]
        if isinstance(last, Mapping):
            return str(last.get("content", ""))
    return ""


def greedy_iou_reward(
    completion: Any, solution: Any, *, iou_threshold: float = 0.5
) -> float:
    """Verifiable IoU reward from GUIDE step 9; malformed JSON receives zero."""
    from scripts.score import iou_xyxy, parse_output

    raw = _completion_text(completion)
    if isinstance(solution, str):
        solution_raw = solution
    else:
        solution_raw = json.dumps(solution, separators=(",", ":"), ensure_ascii=False)
    pred = parse_output(raw)
    truth = parse_output(solution_raw)
    if pred is None or truth is None:
        return 0.0
    if not pred and not truth:
        return 1.0

    candidates = sorted(
        (
            (iou_xyxy(gt["bbox"], pr["bbox"]), gt_i, pr_i)
            for gt_i, gt in enumerate(truth)
            for pr_i, pr in enumerate(pred)
            if gt["label"] == pr["label"]
        ),
        reverse=True,
    )
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    total_iou = 0.0
    for value, gt_i, pred_i in candidates:
        if value < iou_threshold:
            break
        if gt_i in matched_gt or pred_i in matched_pred:
            continue
        matched_gt.add(gt_i)
        matched_pred.add(pred_i)
        total_iou += value
    return 0.2 + 0.8 * total_iou / max(len(truth), len(pred), 1)


def grpo_iou_reward(completions: Sequence[Any], solution: Sequence[Any], **_: Any) -> list[float]:
    """TRL-compatible vector wrapper for :func:`greedy_iou_reward`."""
    if len(completions) != len(solution):
        raise ValueError("TRL must pass one ground-truth solution per completion")
    return [greedy_iou_reward(c, s) for c, s in zip(completions, solution)]


def load_qwen3vl(model_id: str = MODEL_ID) -> tuple[Any, Any]:
    """Load the lab's NF4 model on one CUDA GPU, leaving vision/lm_head in fp16."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL training/inference requires a CUDA GPU; select Colab T4")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    if getattr(processor.tokenizer, "pad_token_id", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor


def _model_device(model: Any):
    import torch

    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, RuntimeError):
        return next(model.parameters()).device


def _find_visual_module(model: Any) -> Any | None:
    """Find Qwen3-VL's vision tower across base and PEFT wrapper layers."""
    pending = [model]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        visual = getattr(candidate, "visual", None)
        if visual is not None:
            return visual
        nested = getattr(candidate, "model", None)
        if nested is not None and nested is not candidate:
            pending.append(nested)
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                base_model = get_base_model()
            except (AttributeError, RuntimeError):
                base_model = None
            if base_model is not candidate:
                pending.append(base_model)
    return None


def _move_model_inputs(encoded: Mapping[str, Any], model: Any) -> dict[str, Any]:
    import torch

    device = _model_device(model)
    visual_dtype = None
    visual = _find_visual_module(model)
    if visual is not None:
        visual_dtype = next(visual.parameters()).dtype
    moved: dict[str, Any] = {}
    for key, value in encoded.items():
        if not torch.is_tensor(value):
            continue
        value = value.to(device)
        if key == "pixel_values" and visual_dtype is not None:
            value = value.to(dtype=visual_dtype)
        moved[key] = value
    return moved


def generate_predictions(
    model: Any,
    processor: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    data_root: str | Path = "data",
    output_path: str | Path,
    coord: str = "norm1000",
    batch_size: int = 4,
    max_new_tokens: int = 256,
) -> list[dict[str, str]]:
    """Run deterministic batched visual grounding and write scorer JSONL."""
    import torch
    from PIL import Image

    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    if coord not in PROMPTS:
        raise ValueError(f"unsupported coordinate system: {coord!r}")
    processor.tokenizer.padding_side = "left"
    model.eval()
    results: list[dict[str, str]] = []
    root = Path(data_root)
    prompt = PROMPTS[coord]
    for offset in range(0, len(rows), batch_size):
        selected = rows[offset : offset + batch_size]
        images = []
        conversations = []
        for row in selected:
            with Image.open(root / row["image_path"]) as source:
                image = source.convert("RGB")
            images.append(image)
            conversations.append(build_messages(image, prompt))
        encoded = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = _move_model_inputs(encoded, model)
        prompt_width = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        completions = processor.batch_decode(
            generated[:, prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        results.extend(
            {"image_id": row["image_id"], "raw": raw.strip()}
            for row, raw in zip(selected, completions)
        )
        if len(results) % max(batch_size * 10, 1) == 0 or len(results) == len(rows):
            print(f"generated {len(results)}/{len(rows)} predictions")
        del encoded, inputs, generated, images, conversations
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return results


def save_prediction_previews(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    data_root: str | Path = "data",
    output_dir: str | Path = "report/figures/predictions",
    coord: str = "norm1000",
    count: int = 6,
    seed: int = 42,
) -> list[Path]:
    """Save deterministic GT/prediction overlays for qualitative error review."""
    from PIL import Image, ImageDraw
    from scripts.score import parse_output

    if coord not in PROMPTS:
        raise ValueError(f"unsupported coordinate system: {coord!r}")
    if count < 1:
        raise ValueError("count must be positive")
    prediction_by_id = {str(row["image_id"]): row.get("raw", "") for row in predictions}
    selected = sorted(
        random.Random(seed).sample(list(rows), min(count, len(rows))),
        key=lambda row: row["image_id"],
    )
    root, out = Path(data_root), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    class_colors = {"pothole": "#ff4d4f", "crack": "#ffd43b", "manhole": "#339af0"}
    written: list[Path] = []
    for row in selected:
        with Image.open(root / row["image_path"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for obj in row.get("objects", []):
            xc, yc, box_w, box_h = obj["yolo"]
            gt_box = (
                round((xc - box_w / 2) * image.width),
                round((yc - box_h / 2) * image.height),
                round((xc + box_w / 2) * image.width),
                round((yc + box_h / 2) * image.height),
            )
            draw.rectangle(gt_box, outline="#37b24d", width=2)
        parsed = parse_output(str(prediction_by_id.get(str(row["image_id"]), "")))
        if parsed is not None:
            x_scale, y_scale = _coord_limits(coord, image.width, image.height)
            for obj in parsed:
                coords = obj["bbox"]
                pred_box = (
                    round(coords[0] / x_scale * image.width),
                    round(coords[1] / y_scale * image.height),
                    round(coords[2] / x_scale * image.width),
                    round(coords[3] / y_scale * image.height),
                )
                draw.rectangle(pred_box, outline=class_colors.get(obj["label"], "#f06595"), width=2)
        else:
            draw.text((4, 4), "invalid / missing prediction", fill="#ffffff")
        image_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["image_id"]))
        target = out / f"{image_id}.png"
        image.save(target)
        written.append(target)
    return written


def apply_setup(model: Any, setup: str, *, rank: int = 16, alpha: int | None = None) -> tuple[Any, dict[str, Any]]:
    """Freeze base weights, attach the specified LoRA targets, and open E2/E3 mergers."""
    import torch
    from peft import LoraConfig, get_peft_model

    setup = setup.upper()
    if setup not in {"E1", "E2", "E3"}:
        raise ValueError("setup must be E1, E2, or E3")
    alpha = 2 * rank if alpha is None else alpha
    for parameter in model.parameters():
        parameter.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    module_names = [name for name, _ in model.named_modules()]
    targets = find_lora_target_names(module_names, setup)
    expected = 348 if setup == "E3" else 252
    if len(targets) != expected:
        sample = [name for name in module_names if "language_model.layers" in name or "visual.blocks" in name][:20]
        raise RuntimeError(
            f"{setup} expected {expected} LoRA target modules, found {len(targets)}. "
            f"Inspect model module names before training. Sample: {sample}"
        )
    wrapped = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    projector_params = []
    if setup in {"E2", "E3"}:
        for name, parameter in wrapped.named_parameters():
            if "visual.merger" in name or "visual.deepstack_merger_list" in name:
                parameter.data = parameter.data.to(dtype=torch.float32)
                parameter.requires_grad = True
                projector_params.append(name)
        if not projector_params:
            raise RuntimeError("Qwen3-VL projector parameters were not found")
    trainable = sum(parameter.numel() for parameter in wrapped.parameters() if parameter.requires_grad)
    lora_modules = sum(
        1
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad and "lora_A" in name
    )
    if lora_modules != expected:
        raise RuntimeError(f"PEFT attached {lora_modules} LoRA modules; expected {expected}")
    if not hasattr(wrapped, "gradient_checkpointing_enable"):
        raise RuntimeError("model does not support gradient checkpointing")
    wrapped.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    wrapped.config.use_cache = False
    metadata = {
        "setup": setup,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "matched_lora_modules": lora_modules,
        "trainable_params": trainable,
        "trainable_projector_params": sum(
            dict(wrapped.named_parameters())[name].numel() for name in projector_params
        ),
    }
    print(
        f"{setup}: {lora_modules} LoRA modules; "
        f"{trainable:,} trainable parameters; "
        f"projector={metadata['trainable_projector_params']:,}"
    )
    return wrapped, metadata


def _projector_state_dict(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().to(device="cpu", dtype=__import__("torch").float32)
        for name, parameter in model.named_parameters()
        if "visual.merger" in name or "visual.deepstack_merger_list" in name
    }


def save_checkpoint(
    model: Any,
    destination: str | Path,
    *,
    setup: str,
    metadata: Mapping[str, Any],
) -> Path:
    """Save LoRA plus full-trained projector weights, if the setup has one."""
    import torch

    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target / "adapter", safe_serialization=True)
    if setup.upper() in {"E2", "E3"}:
        projector = _projector_state_dict(model)
        if not projector:
            raise RuntimeError(f"{setup} has no trainable projector weights to save")
        torch.save(projector, target / "projector.pt")
    (target / "run.json").write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def _load_yaml_config(setup: str, config_path: str | Path | None) -> dict[str, Any]:
    import yaml

    path = Path(config_path) if config_path else Path("configs") / f"{setup.lower()}.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    return config


def train_setup(
    setup: str,
    train_rows: Sequence[Mapping[str, Any]],
    *,
    data_root: str | Path = "data",
    output_root: str | Path = "/content/drive/MyDrive/K4-Day10/checkpoints",
    run_name: str | None = None,
    config_path: str | Path | None = None,
    coord: str = "norm1000",
    train_fraction: float = 1.0,
    rank: int | None = None,
    alpha: int | None = None,
    max_steps: int | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    learning_rate: float | None = None,
    save_every: int = 100,
) -> dict[str, Any]:
    """Run manual fp16 QLoRA SFT with gradient accumulation and Drive checkpoints."""
    import gc
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    setup = setup.upper()
    if setup not in {"E1", "E2", "E3"}:
        raise ValueError("setup must be E1, E2, or E3")
    if coord not in PROMPTS:
        raise ValueError(f"unsupported coordinate system: {coord!r}")
    config = _load_yaml_config(setup, config_path)
    lora = config.get("lora", {})
    seed = int(config.get("seed", 42))
    rank = int(rank if rank is not None else lora.get("r", 16))
    alpha = int(alpha if alpha is not None else lora.get("alpha", rank * 2))
    steps = int(max_steps if max_steps is not None else config.get("max_steps", 300))
    micro_batch = int(batch_size if batch_size is not None else config.get("batch_size", 2))
    accumulation = int(grad_accum if grad_accum is not None else config.get("grad_accum", 4))
    lr = float(learning_rate if learning_rate is not None else config.get("lr", 2e-4))
    if steps < 1 or micro_batch < 1 or accumulation < 1 or save_every < 1:
        raise ValueError("steps, batch_size, grad_accum, and save_every must be positive")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, processor = load_qwen3vl(str(config.get("model_id", MODEL_ID)))
    model, setup_metadata = apply_setup(model, setup, rank=rank, alpha=alpha)
    dataset = RoadDamageDataset(
        train_rows,
        processor,
        data_root=data_root,
        coord=coord,
        fraction=train_fraction,
        seed=seed,
    )
    if len(dataset) < micro_batch:
        raise ValueError(f"train dataset has {len(dataset)} rows, smaller than batch_size={micro_batch}")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda items: collate_examples(
            items, pad_token_id=processor.tokenizer.pad_token_id
        ),
    )
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters found")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)
    warmup_steps = max(1, round(steps * 0.05))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps
    )
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.monotonic()
    run_name = run_name or setup
    run_root = Path(output_root) / run_name
    losses: list[float] = []
    skipped_updates = 0
    successful_steps = 0
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)

    while successful_steps < steps:
        accumulated_losses = []
        for _ in range(accumulation):
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                cpu_batch = next(iterator)
            batch = _move_model_inputs(cpu_batch, model)
            batch.pop("image_id", None)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss
            if loss is None or not bool(torch.isfinite(loss).item()):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"non-finite loss at optimizer step {successful_steps + 1}; "
                    "stop and inspect fp16 stability and label masking"
                )
            accumulated_losses.append(float(loss.detach().item()))
            scaler.scale(loss / accumulation).backward()
            del batch, cpu_batch, loss

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        if not bool(torch.isfinite(grad_norm).item()):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            skipped_updates += 1
            if skipped_updates >= 5:
                raise FloatingPointError("five consecutive non-finite gradient updates; stopping")
            print(f"non-finite gradient norm; skipped update {skipped_updates}/5")
            continue
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scale_after = scaler.get_scale()
        if scale_after < scale_before:
            skipped_updates += 1
            if skipped_updates >= 5:
                raise FloatingPointError("five consecutive fp16 overflow updates; stopping")
            print(f"GradScaler skipped overflow update {skipped_updates}/5")
            continue

        scheduler.step()
        successful_steps += 1
        step_loss = sum(accumulated_losses) / len(accumulated_losses)
        losses.append(step_loss)
        if successful_steps == 1 or successful_steps % 10 == 0:
            print(
                f"{run_name} step {successful_steps}/{steps} "
                f"loss={step_loss:.4f} grad_norm={float(grad_norm):.3f} "
                f"lr={scheduler.get_last_lr()[0]:.3g}"
            )

        elapsed_min = (time.monotonic() - start_time) / 60.0
        metadata = {
            **setup_metadata,
            "model_id": str(config.get("model_id", MODEL_ID)),
            "coord": coord,
            "train_fraction": train_fraction,
            "seed": seed,
            "batch_size": micro_batch,
            "grad_accum": accumulation,
            "effective_batch_size": micro_batch * accumulation,
            "max_steps": steps,
            "successful_steps": successful_steps,
            "skipped_updates": skipped_updates,
            "learning_rate": lr,
            "warmup_steps": warmup_steps,
            "trainable_params": setup_metadata["trainable_params"],
            "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "train_minutes": elapsed_min,
            "loss_first_10_avg": sum(losses[:10]) / min(len(losses), 10),
            "loss_last_10_avg": sum(losses[-10:]) / min(len(losses), 10),
        }
        if successful_steps % save_every == 0:
            checkpoint = run_root / "checkpoints" / f"step-{successful_steps:04d}"
            save_checkpoint(model, checkpoint, setup=setup, metadata=metadata)

    final_metadata = {
        **metadata,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "train_minutes": (time.monotonic() - start_time) / 60.0,
    }
    save_checkpoint(model, run_root / "final", setup=setup, metadata=final_metadata)
    model.eval()
    del model, processor, optimizer, scheduler, scaler, trainable_params, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return final_metadata


def load_trained_setup(
    setup: str,
    *,
    output_root: str | Path = "/content/drive/MyDrive/K4-Day10/checkpoints",
    run_name: str | None = None,
    model_id: str = MODEL_ID,
) -> tuple[Any, Any]:
    """Load a trained adapter and restore E2/E3 full-precision projector weights."""
    import torch
    from peft import PeftModel

    setup = setup.upper()
    if setup not in {"E1", "E2", "E3"}:
        raise ValueError("setup must be E1, E2, or E3")
    run_root = Path(output_root) / (run_name or setup) / "final"
    model, processor = load_qwen3vl(model_id)
    model = PeftModel.from_pretrained(model, run_root / "adapter", is_trainable=False)
    if setup in {"E2", "E3"}:
        parameters = dict(model.named_parameters())
        for name, parameter in parameters.items():
            if "visual.merger" in name or "visual.deepstack_merger_list" in name:
                parameter.data = parameter.data.to(dtype=torch.float32)
        weights = torch.load(run_root / "projector.pt", map_location="cpu", weights_only=True)
        missing = set(weights) - set(parameters)
        if missing:
            raise RuntimeError(f"saved projector keys are absent from this model: {sorted(missing)[:5]}")
        with torch.no_grad():
            for name, value in weights.items():
                parameters[name].copy_(value.to(device=parameters[name].device, dtype=torch.float32))
    model.eval()
    return model, processor

