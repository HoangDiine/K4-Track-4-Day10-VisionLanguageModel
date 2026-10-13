# Day 10 solution notes

## Files added or updated

- `lab_solution.py` contains the student-owned implementations for YOLO-to-grounding labels, chat-template encoding and answer-only loss masking, batching multimodal inputs, QLoRA setup, SFT training, checkpoint restore, deterministic inference, prediction visualizations, and the tested IoU reward.
- `lab_solution_selftest.py` checks those helpers without downloading the model or dataset.
- `lab_day10.ipynb` is a self-contained Colab workflow. It writes the two solution files into the cloned repo, runs the instructor tests and the helper tests, prepares the dataset, and provides gated cells for E0–E4.
- `requirements-colab.txt` pins `torchao==0.17.0` to avoid the incompatible `0.10.0` build bundled by the current Colab runtime with PEFT 0.21.2.

## Run order

1. Open `lab_day10.ipynb` in Colab, select a T4 runtime, and run the environment cells. They report the Python and package versions and run both test suites.
2. Mount Drive. The public Kaggle dataset is downloaded through Kaggle's unauthenticated dataset endpoint when no `DATA_SOURCE`, `DATA_ZIP`, or Kaggle Secrets are provided. Secrets remain optional, and the notebook does not print their values.
3. Prepare and inspect the data. The split helper checks session isolation and saves six ground-truth previews.
4. Run the chat-template and target-module checks. For a 640×360 input, the processor should produce 220 image tokens; E1/E2 use 252 LoRA modules and E3 uses 348.
5. Turn on `RUN_E0` to generate and score the baseline. Turn on `RUN_SFT_EXPERIMENTS` for E1–E3, then `RUN_EVAL_TRAINED` to score their checkpoints and a fixed 40-image validation sample. E4 chooses its base setup from validation F1, then evaluates the rank ablation on test.

The expensive inference and training flags default to `False`. Enable one stage at a time after the dataset and Drive are ready. Checkpoints are written under `MyDrive/K4-Day10/checkpoints`; predictions, evidence JSON, and six prediction overlays per evaluated setup are copied under `MyDrive/K4-Day10/submission_artifacts`.

## Evidence and report

The notebook does not invent evaluation numbers or fill personal report answers. Run the real experiments, use their `report/evidence/*.json` outputs to answer `REPORT.md`, add the required E4 evidence, and then run `python scripts/check_submission.py`. E5 is optional; its IoU reward is implemented and covered by the helper tests, while GRPO training is not part of the mandatory E0–E4 workflow.

Keep `data/`, `work/`, credentials, checkpoints, and model weights out of Git. The instructor-provided `scripts/`, `tests/`, and `configs/` remain unchanged.
