"""Fast, data-free checks for the student implementation.

Run with: python -m pytest -q lab_solution_selftest.py
This file is intentionally outside the instructor-provided tests/ directory.
"""

import json

import pytest

from lab_solution import (
    collate_examples,
    find_lora_target_names,
    greedy_iou_reward,
    objects_to_answer,
    save_prediction_previews,
    yolo_to_box,
)


def test_yolo_box_is_integer_clamped_and_round_trips():
    box = yolo_to_box("pothole", [0.5, 0.5, 0.4, 0.2])
    assert box == {"label": "pothole", "bbox_2d": [300, 400, 700, 600]}

    edge = yolo_to_box("crack", [0.0, 1.0, 0.4, 0.4])
    assert edge == {"label": "crack", "bbox_2d": [0, 800, 200, 1000]}
    assert all(isinstance(coord, int) and 0 <= coord <= 1000 for coord in edge["bbox_2d"])


def test_pixel_coordinates_use_image_dimensions():
    box = yolo_to_box("manhole", [0.5, 0.5, 0.5, 0.5], width=640, height=360, coord="pixel")
    assert box == {"label": "manhole", "bbox_2d": [160, 90, 480, 270]}


def test_objects_to_answer_is_stable_and_empty_is_json_list():
    objects = [
        {"label": "crack", "yolo": [0.7, 0.7, 0.2, 0.2]},
        {"label": "pothole", "yolo": [0.3, 0.3, 0.2, 0.2]},
        {"label": "manhole", "yolo": [0.7, 0.3, 0.2, 0.2]},
    ]
    answer = objects_to_answer(objects)
    parsed = json.loads(answer)
    assert [box["label"] for box in parsed] == ["pothole", "manhole", "crack"]
    assert answer == objects_to_answer(objects)
    assert objects_to_answer([]) == "[]"


def test_reward_covers_perfect_invalid_wrong_class_and_empty_cases():
    ground_truth = '[{"label":"pothole","bbox_2d":[100,100,300,300]}]'
    perfect = '[{"label":"pothole","bbox_2d":[100,100,300,300]}]'
    wrong_class = '[{"label":"crack","bbox_2d":[100,100,300,300]}]'

    assert greedy_iou_reward(perfect, ground_truth) == pytest.approx(1.0)
    assert greedy_iou_reward("not json", ground_truth) == 0.0
    assert greedy_iou_reward(wrong_class, ground_truth) == pytest.approx(0.2)
    assert greedy_iou_reward("[]", "[]") == pytest.approx(1.0)


def test_reward_penalizes_extra_boxes_through_denominator():
    ground_truth = '[{"label":"pothole","bbox_2d":[100,100,300,300]}]'
    two_preds = (
        '[{"label":"pothole","bbox_2d":[100,100,300,300]},'
        '{"label":"crack","bbox_2d":[400,400,600,600]}]'
    )
    assert greedy_iou_reward(two_preds, ground_truth) == pytest.approx(0.6)


def test_lora_target_counts_match_qwen3_vl_lab_contract():
    names = []
    for layer in range(36):
        names.extend(
            f"model.language_model.layers.{layer}.self_attn.{name}"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        names.extend(
            f"model.language_model.layers.{layer}.mlp.{name}"
            for name in ("gate_proj", "up_proj", "down_proj")
        )
    for block in range(24):
        names.extend(
            f"model.visual.blocks.{block}.{name}"
            for name in (
                "attn.qkv",
                "attn.proj",
                "mlp.linear_fc1",
                "mlp.linear_fc2",
            )
        )

    assert len(find_lora_target_names(names, "E1")) == 252
    assert len(find_lora_target_names(names, "E2")) == 252
    assert len(find_lora_target_names(names, "E3")) == 348


def test_collator_pads_text_and_masks_labels_but_concatenates_images():
    torch = pytest.importorskip("torch")
    batch = [
        {
            "input_ids": torch.tensor([[10, 11]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 1]]),
            "labels": torch.tensor([[-100, 11]]),
            "pixel_values": torch.ones((2, 3)),
            "image_grid_thw": torch.tensor([[1, 2, 1]]),
        },
        {
            "input_ids": torch.tensor([[20, 21, 22]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 0, 1]]),
            "labels": torch.tensor([[-100, -100, 22]]),
            "pixel_values": torch.ones((3, 3)) * 2,
            "image_grid_thw": torch.tensor([[1, 3, 1]]),
        },
    ]

    result = collate_examples(batch, pad_token_id=0)
    assert result["input_ids"].tolist() == [[10, 11, 0], [20, 21, 22]]
    assert result["attention_mask"].tolist() == [[1, 1, 0], [1, 1, 1]]
    assert result["mm_token_type_ids"].tolist() == [[0, 1, 0], [0, 0, 1]]
    assert result["labels"].tolist() == [[-100, 11, -100], [-100, -100, 22]]
    assert tuple(result["pixel_values"].shape) == (5, 3)
    assert tuple(result["image_grid_thw"].shape) == (2, 3)


def test_prediction_previews_save_requested_images(tmp_path):
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (64, 36), "black").save(image_dir / "sample.jpg")
    row = {
        "image_id": "sample",
        "image_path": "images/sample.jpg",
        "width": 64,
        "height": 36,
        "objects": [],
    }
    output = save_prediction_previews(
        [row],
        [{"image_id": "sample", "raw": '[{"label":"pothole","bbox_2d":[100,100,900,900]}]'}],
        data_root=tmp_path,
        output_dir=tmp_path / "previews",
        coord="norm1000",
        count=1,
    )
    assert len(output) == 1
    assert output[0].is_file()
    assert Image.open(output[0]).size == (64, 36)
