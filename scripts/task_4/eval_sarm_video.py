import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from decord import VideoReader, cpu

from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel
from lerobot.rewards.sarm.processor_sarm import make_sarm_pre_post_processors
from lerobot.rewards.sarm.sarm_utils import compute_absolute_indices


TASKS = {
    0: "open the middle drawer of the cabinet",
    1: "put the bowl on the stove",
    2: "put the wine bottle on top of the cabinet",
}


def load_video(video_path: Path) -> np.ndarray:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    return vr.get_batch(range(len(vr))).asnumpy()


@torch.no_grad()
def evaluate_video(
    video_path: Path,
    task: str,
    reward_model: SARMRewardModel,
    preprocess,
) -> np.ndarray:

    frames = load_video(video_path)
    num_frames = len(frames)

    config = reward_model.config
    image_key = config.image_key
    state_key = config.state_key

    center_idx = config.n_obs_steps // 2

    rewards = np.zeros(num_frames, dtype=np.float32)

    for t in range(num_frames):

        indices, _ = compute_absolute_indices(
            frame_idx=t,
            ep_start=0,
            ep_end=num_frames,
            n_obs_steps=config.n_obs_steps,
            frame_gap=config.frame_gap,
        )

        window = frames[indices.numpy()]

        window = (
            torch.from_numpy(window)
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )

        state = torch.zeros(
            window.shape[0],
            config.max_state_dim,
            dtype=torch.float32,
        )

        batch = {
            image_key: window,
            state_key: state,
            "task": task,
            "index": t,
            "episode_index": 0,
        }

        processed = preprocess(batch)

        progress = reward_model.calculate_rewards(
            text_embeddings=processed["text_features"],
            video_embeddings=processed["video_features"],
            state_features=processed["state_features"],
            lengths=processed["lengths"],
            return_all_frames=True,
            head_mode="sparse",
        )

        rewards[t] = float(progress[0, center_idx])

    return rewards


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--reward-model-path", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--task-id", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    task = TASKS[args.task_id]

    reward_model = SARMRewardModel.from_pretrained(
        args.reward_model_path
    )

    reward_model.config.device = args.device
    reward_model.to(args.device).eval()

    preprocess, _ = make_sarm_pre_post_processors(
        config=reward_model.config,
        dataset_stats=None,
        dataset_meta=None,
    )

    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()

    videos = sorted(video_dir.glob("eval_episode_*.mp4"))

    if not videos:
        raise RuntimeError(f"No videos found in {video_dir}")

    for video_path in videos:

        rewards = evaluate_video(
            video_path=video_path,
            task=task,
            reward_model=reward_model,
            preprocess=preprocess,
        )

        npy_path = video_path.with_name(
            video_path.stem + "_sarm_rewards.npy"
        )

        np.save(npy_path, rewards)

        png_path = video_path.with_name(
            video_path.stem + "_sarm_progress.png"
        )

        plt.figure(figsize=(10, 4))
        plt.plot(rewards)
        plt.ylim(0, 1)
        plt.xlabel("Frame")
        plt.ylabel("SARM progress")
        plt.title(video_path.name)
        plt.grid()
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

        print(
            f"{video_path.name}: "
            f"first={rewards[0]:.3f}, "
            f"final={rewards[-1]:.3f}, "
            f"max={rewards.max():.3f}, "
            f"mean={rewards.mean():.3f}"
        )


if __name__ == "__main__":
    main()