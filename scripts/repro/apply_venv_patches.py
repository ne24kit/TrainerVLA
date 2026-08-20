#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".repro.bak")
    if not backup.exists():
        backup.write_text(read(path), encoding="utf-8")


def replace_once(path: Path, old: str, new: str, patch_name: str) -> str:
    text = read(path)
    if new in text:
        return "already applied"
    if old not in text:
        raise RuntimeError(f"{patch_name}: anchor not found in {path}")
    backup_once(path)
    write(path, text.replace(old, new, 1))
    return "applied"


def lerobot_root() -> Path:
    spec = importlib.util.find_spec("lerobot")
    if spec is None or spec.origin is None:
        raise RuntimeError("lerobot is not installed in this Python environment")
    return Path(spec.origin).resolve().parent


def patch_eval_gripper(site: Path) -> str:
    path = site / "scripts" / "lerobot_eval.py"
    marker = "## MY PATCH FOR SMOLVLA POST PROCESSING"
    text = read(path)
    if marker in text:
        return "already applied"

    old = """            action_transition = {ACTION: action}
            action_transition = env_postprocessor(action_transition)
            action = action_transition[ACTION]

            # Convert to CPU / numpy.
"""
    new = """            action_transition = {ACTION: action}
            action_transition = env_postprocessor(action_transition)
            action = action_transition[ACTION]

            ## MY PATCH FOR SMOLVLA POST PROCESSING
            if action.shape[-1] >= 7:
                action = action.clone()
                gripper = action[..., 6]
                action[..., 6] = 1.0 - 2.0 * (gripper > 0.5).to(action.dtype)

            # Convert to CPU / numpy.
"""
    return replace_once(path, old, new, "eval gripper binarization")


def patch_sarm_zero_state(site: Path) -> str:
    path = site / "rewards" / "sarm" / "processor_sarm.py"
    marker = "ZERO-STATE ABLATION"
    text = read(path)
    if marker in text:
        return "already applied"

    old = """        if isinstance(state_data, torch.Tensor):
            state_tensor = state_data.float()
        else:
            state_tensor = torch.tensor(state_data, dtype=torch.float32)

"""
    new = """        if isinstance(state_data, torch.Tensor):
            state_tensor = state_data.float()
        else:
            state_tensor = torch.tensor(state_data, dtype=torch.float32)

        # ZERO-STATE ABLATION:
        # Do not allow SARM to use robot state information.
        state_tensor = torch.zeros_like(state_tensor)

"""
    return replace_once(path, old, new, "SARM zero-state ablation")


def patch_libero_render_fps(site: Path) -> str:
    path = site / "envs" / "libero.py"
    text = read(path)
    if '"render_fps": 20' in text:
        return "already applied"
    old = 'metadata = {"render_modes": ["rgb_array"], "render_fps": 80}'
    new = 'metadata = {"render_modes": ["rgb_array"], "render_fps": 20}'
    return replace_once(path, old, new, "LIBERO render fps")


def patch_robometer_mm_token_type_ids() -> str:
    path = ROOT / "robometer" / "robometer" / "evals" / "eval_server.py"
    if not path.exists():
        return "skipped: robometer/ is absent"

    text = read(path)
    if "mm_token_type_ids=batch_inputs.get" in text:
        return "already applied"

    old = """                video_grid_thw=batch_inputs.get("video_grid_thw", None),
                second_per_grid_ts=batch_inputs.get("second_per_grid_ts", None),
"""
    new = """                video_grid_thw=batch_inputs.get("video_grid_thw", None),

                # Required by Qwen3-VL / transformers 5.5+
                mm_token_type_ids=batch_inputs.get("mm_token_type_ids", None),

                second_per_grid_ts=batch_inputs.get("second_per_grid_ts", None),
"""
    return replace_once(path, old, new, "Robometer mm_token_type_ids")


def verify_pipeline_state_file_support(site: Path) -> str:
    path = site / "processor" / "pipeline.py"
    text = read(path)
    required = [
        "_serialized_state_filenames",
        "_get_state_filenames_from_config",
        'step_entry["state_file"]',
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(
            "lerobot.processor.pipeline.py does not expose the processor state_file "
            f"logic needed by these checkpoints. Missing: {missing}"
        )
    return "ok"


def main() -> None:
    site = lerobot_root()
    results = {
        "lerobot_eval gripper patch": patch_eval_gripper(site),
        "SARM zero-state patch": patch_sarm_zero_state(site),
        "LIBERO render fps patch": patch_libero_render_fps(site),
        "Robometer Qwen3 patch": patch_robometer_mm_token_type_ids(),
        "processor state_file support": verify_pipeline_state_file_support(site),
    }

    for name, status in results.items():
        print(f"{name}: {status}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
