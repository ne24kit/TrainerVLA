import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run", required=True)
    parser.add_argument("--log-freq", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)

    parser.add_argument(
        "--base-dir",
        default="/home/msi/projects/TrainerVLA",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_dir = Path(args.base_dir) / "logs" / args.run
    log_file = run_dir / "train.log"
    plot_dir = run_dir / "plots"

    if not log_file.exists():
        raise FileNotFoundError(log_file)

    plot_dir.mkdir(parents=True, exist_ok=True)

    train_loss = []

    val_steps = []
    val_loss = []

    success_steps = []
    success_values = []

    current_eval_step = None

    with open(log_file, "r", errors="ignore") as f:
        for line in f:

            # -------------------------
            # TRAIN LOSS
            # -------------------------

            match = re.search(
                r"(?:^|\s)loss:([+-]?\d*\.?\d+(?:e[+-]?\d+)?)",
                line,
                re.IGNORECASE,
            )

            if match:
                train_loss.append(float(match.group(1)))

            # -------------------------
            # VALIDATION LOSS
            # step 1000: eval_loss=...
            # -------------------------

            match = re.search(
                r"step\s+(\d+):\s+eval_loss=([+-]?\d*\.?\d+(?:e[+-]?\d+)?)",
                line,
                re.IGNORECASE,
            )

            if match:
                val_steps.append(int(match.group(1)))
                val_loss.append(float(match.group(2)))

            # -------------------------
            # ENV EVAL START
            # -------------------------

            match = re.search(
                r"Eval policy at step\s+(\d+)",
                line,
                re.IGNORECASE,
            )

            if match:
                current_eval_step = int(match.group(1))

            # -------------------------
            # LIBERO SUCCESS
            # -------------------------

            if current_eval_step is not None and "pc_success" in line:

                match = re.search(
                    r"""['"]pc_success['"]\s*:\s*([0-9.]+)""",
                    line,
                )

                if match:
                    success_steps.append(current_eval_step)
                    success_values.append(float(match.group(1)))
                    current_eval_step = None

    # Exact train steps reconstructed from log_freq
    train_steps = [
        (i + 1) * args.log_freq
        for i in range(len(train_loss))
    ]

    # ==========================
    # LOSS
    # ==========================

    if train_loss or val_loss:

        plt.figure(figsize=(10, 5))

        if train_loss:
            plt.plot(
                train_steps,
                train_loss,
                label="Train",
            )

        if val_loss:
            plt.plot(
                val_steps,
                val_loss,
                marker="o",
                label="Validation",
            )

        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("SmolVLA loss")

        plt.xlim(
            0,
            max(
                train_steps[-1] if train_steps else 0,
                val_steps[-1] if val_steps else 0,
            ),
        )

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            plot_dir / "loss.png",
            dpi=150,
        )

        plt.close()

    # ==========================
    # SUCCESS
    # ==========================

    if success_values:

        plt.figure(figsize=(10, 5))

        plt.plot(
            success_steps,
            success_values,
            marker="o",
        )

        plt.xlabel("Training step")
        plt.ylabel("Success rate, %")
        plt.title("LIBERO seen-task success")

        plt.xlim(0, success_steps[-1])
        plt.ylim(0, 100)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            plot_dir / "success.png",
            dpi=150,
        )

        plt.close()

    print(f"\nLog:   {log_file}")
    print(f"Plots: {plot_dir}")

    print(f"\nTrain loss points: {len(train_loss)}")
    print(f"Validation points: {len(val_loss)}")
    print(f"Success points:    {len(success_values)}")

    if train_loss:
        print(f"Last train loss:   {train_loss[-1]:.4f}")

    if val_loss:
        print(f"Last val loss:     {val_loss[-1]:.4f}")

    if success_values:
        print(f"Last success:      {success_values[-1]:.1f}%")


if __name__ == "__main__":
    main()