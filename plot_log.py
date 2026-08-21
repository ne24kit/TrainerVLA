import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trainervla_matplotlib")

import matplotlib.pyplot as plt


FLOAT_RE = r"[+-]?\d*\.?\d+(?:e[+-]?\d+)?"
TRAIN_LOSS_RE = re.compile(rf"(?:^|\s)loss:({FLOAT_RE})", re.IGNORECASE)
TRAIN_STEP_RE = re.compile(r"(?:^|\s)step:(\d+(?:\.\d+)?)([KMB]?)\b", re.IGNORECASE)
PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s+\[")
CFG_STEPS_RE = re.compile(r"cfg\.steps=(\d+)|['\"]steps['\"]:\s*(\d+)")
VAL_LOSS_RE = re.compile(rf"step\s+(\d+):\s+eval_loss=({FLOAT_RE})", re.IGNORECASE)
EVAL_STEP_RE = re.compile(r"Eval policy at step\s+(\d+)", re.IGNORECASE)
SUCCESS_RE = re.compile(r"""['"]pc_success['"]\s*:\s*([0-9.]+)""")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run", default=None)
    parser.add_argument("--log-freq", type=int, default=None, help="Fallback only for old logs.")
    parser.add_argument("--steps", type=int, default=None, help="Fallback total steps.")
    parser.add_argument(
        "--log-filename",
        "--log_filename",
        dest="log_filenames",
        action="append",
        help="Log file inside logs/<run>. Can be passed more than once. Defaults to train.log + train_resume_*.log.",
    )

    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parent),
    )

    return parser.parse_args()


def parse_compact_int(value: str, suffix: str) -> int:
    multiplier = {
        "": 1,
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
    }[suffix.upper()]
    return int(float(value) * multiplier)


def parse_cfg_steps(line: str) -> int | None:
    match = CFG_STEPS_RE.search(line)
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def parse_train_step(line: str, total_steps: int | None) -> tuple[int | None, int | None]:
    progress_matches = list(PROGRESS_RE.finditer(line))
    if progress_matches:
        progress = progress_matches[-1]
        local_step = int(progress.group(1))
        progress_total = int(progress.group(2))
        if total_steps is not None:
            run_start_step = total_steps - progress_total
            return run_start_step + local_step, run_start_step

    match = TRAIN_STEP_RE.search(line)
    if match:
        return parse_compact_int(match.group(1), match.group(2)), None

    return None, None


def discover_log_files(run_dir: Path, log_filenames: list[str] | None) -> list[Path]:
    if log_filenames:
        log_files = [run_dir / name for name in log_filenames]
        missing = [path for path in log_files if not path.exists()]
        if missing:
            raise FileNotFoundError(", ".join(str(path) for path in missing))
        return log_files

    log_files = []
    train_log = run_dir / "train.log"
    if train_log.exists():
        log_files.append(train_log)
    log_files.extend(sorted(run_dir.glob("train_resume_*.log")))

    if not log_files:
        raise FileNotFoundError(f"No train logs found in {run_dir}")

    return log_files


def discover_run_dirs(logs_dir: Path, run: str | None) -> list[Path]:
    if run:
        run_dir = logs_dir / run
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        return [run_dir]

    run_dirs = [
        path
        for path in sorted(logs_dir.iterdir())
        if path.is_dir() and ((path / "train.log").exists() or list(path.glob("train_resume_*.log")))
    ]

    if not run_dirs:
        raise FileNotFoundError(f"No train runs found in {logs_dir}")

    return run_dirs


def dedupe_series(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    by_step = {}
    for step, value in points:
        by_step[step] = value
    return sorted(by_step.items())


def trim_after_resume(points: list[tuple[int, float]], resume_step: int | None) -> list[tuple[int, float]]:
    if resume_step is None or resume_step <= 0:
        return points
    return [(step, value) for step, value in points if step <= resume_step]


def parse_log_file(log_file: Path, fallback_steps: int | None, fallback_log_freq: int | None):
    train_loss = []
    val_loss = []
    success = []

    total_steps = fallback_steps
    resume_step = None
    fallback_train_step = 0
    current_eval_step = None

    with open(log_file, "r", errors="ignore") as f:
        for line in f:
            parsed_steps = parse_cfg_steps(line)
            if parsed_steps is not None:
                total_steps = parsed_steps

            train_step, line_resume_step = parse_train_step(line, total_steps)
            if line_resume_step is not None and line_resume_step > 0:
                resume_step = line_resume_step

            for match in TRAIN_LOSS_RE.finditer(line):
                if train_step is None and fallback_log_freq is not None:
                    fallback_train_step += fallback_log_freq
                    point_step = fallback_train_step
                elif train_step is not None:
                    point_step = train_step
                else:
                    continue

                train_loss.append((point_step, float(match.group(1))))

            for match in VAL_LOSS_RE.finditer(line):
                val_loss.append((int(match.group(1)), float(match.group(2))))

            match = EVAL_STEP_RE.search(line)
            if match:
                current_eval_step = int(match.group(1))

            if current_eval_step is not None and "pc_success" in line:
                success_match = SUCCESS_RE.search(line)
                if success_match and ("Suite overall aggregated" in line or "Suite per_group aggregated" in line):
                    success.append((current_eval_step, float(success_match.group(1))))
                    current_eval_step = None

    return {
        "log_file": log_file,
        "resume_step": resume_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "success": success,
    }


def merge_parsed_logs(parsed_logs):
    train_loss = []
    val_loss = []
    success = []

    for parsed in parsed_logs:
        resume_step = parsed["resume_step"]
        train_loss = trim_after_resume(train_loss, resume_step)
        val_loss = trim_after_resume(val_loss, resume_step)
        success = trim_after_resume(success, resume_step)

        train_loss.extend(parsed["train_loss"])
        val_loss.extend(parsed["val_loss"])
        success.extend(parsed["success"])

    return {
        "train_loss": dedupe_series(train_loss),
        "val_loss": dedupe_series(val_loss),
        "success": dedupe_series(success),
    }


def split_series(points: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
    if not points:
        return [], []
    steps, values = zip(*points)
    return list(steps), list(values)


def save_loss_plot(plot_dir: Path, run: str, train_points, val_points):
    train_steps, train_loss = split_series(train_points)
    val_steps, val_loss = split_series(val_points)

    if not train_loss and not val_loss:
        return

    plt.figure(figsize=(10, 5))

    if train_loss:
        plt.plot(train_steps, train_loss, label="Train")

    if val_loss:
        plt.plot(val_steps, val_loss, marker="o", label="Validation")

    max_step = max(train_steps[-1] if train_steps else 0, val_steps[-1] if val_steps else 0)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"{run} loss")
    plt.xlim(0, max_step)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "loss.png", dpi=150)
    plt.close()


def save_success_plot(plot_dir: Path, run: str, success_points):
    success_steps, success_values = split_series(success_points)

    if not success_values:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(success_steps, success_values, marker="o")
    plt.xlabel("Training step")
    plt.ylabel("Success rate, %")
    plt.title(f"{run} LIBERO seen-task success")
    plt.xlim(0, success_steps[-1])
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "success.png", dpi=150)
    plt.close()


def save_combined_loss_plot(plot_dir: Path, run_data):
    has_any = any(data["train_loss"] or data["val_loss"] for data in run_data)
    if not has_any:
        return

    plt.figure(figsize=(12, 6))
    max_step = 0

    for data in run_data:
        run = data["run"]
        train_steps, train_loss = split_series(data["train_loss"])
        val_steps, val_loss = split_series(data["val_loss"])

        if train_loss:
            line = plt.plot(train_steps, train_loss, label=f"{run} train")[0]
            max_step = max(max_step, train_steps[-1])
            color = line.get_color()
        else:
            color = None

        if val_loss:
            plt.plot(
                val_steps,
                val_loss,
                linestyle="none",
                marker="o",
                markersize=4,
                color=color,
                label=f"{run} val",
            )
            max_step = max(max_step, val_steps[-1])

    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("All runs loss")
    plt.xlim(0, max_step)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "loss.png", dpi=150)
    plt.close()


def save_combined_success_plot(plot_dir: Path, run_data):
    has_any = any(data["success"] for data in run_data)
    if not has_any:
        return

    plt.figure(figsize=(12, 6))
    max_step = 0

    for data in run_data:
        success_steps, success_values = split_series(data["success"])
        if not success_values:
            continue

        plt.plot(success_steps, success_values, marker="o", label=data["run"])
        max_step = max(max_step, success_steps[-1])

    plt.xlabel("Training step")
    plt.ylabel("Success rate, %")
    plt.title("All runs LIBERO seen-task success")
    plt.xlim(0, max_step)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "success.png", dpi=150)
    plt.close()


def load_run_data(run_dir: Path, args):
    log_files = discover_log_files(run_dir, args.log_filenames)
    parsed_logs = [
        parse_log_file(log_file, args.steps, args.log_freq)
        for log_file in log_files
    ]
    merged = merge_parsed_logs(parsed_logs)
    return {
        "run": run_dir.name,
        "log_files": log_files,
        "train_loss": merged["train_loss"],
        "val_loss": merged["val_loss"],
        "success": merged["success"],
    }


def main():
    args = parse_args()

    logs_dir = Path(args.base_dir) / "logs"
    run_dirs = discover_run_dirs(logs_dir, args.run)
    plot_dir = (run_dirs[0] if args.run else logs_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    run_data = [load_run_data(run_dir, args) for run_dir in run_dirs]

    if args.run:
        data = run_data[0]
        save_loss_plot(plot_dir, data["run"], data["train_loss"], data["val_loss"])
        save_success_plot(plot_dir, data["run"], data["success"])
    else:
        save_combined_loss_plot(plot_dir, run_data)
        save_combined_success_plot(plot_dir, run_data)

    print("\nRuns:")
    for data in run_data:
        print(f"  {data['run']}")
        for log_file in data["log_files"]:
            print(f"    {log_file}")

    print(f"Plots: {plot_dir}")

    print()
    for data in run_data:
        train_points = data["train_loss"]
        val_points = data["val_loss"]
        success_points = data["success"]

        print(f"{data['run']}:")
        print(f"  Train loss points: {len(train_points)}")
        print(f"  Validation points: {len(val_points)}")
        print(f"  Success points:    {len(success_points)}")

        if train_points:
            print(f"  Last train step:   {train_points[-1][0]}")
            print(f"  Last train loss:   {train_points[-1][1]:.4f}")

        if val_points:
            print(f"  Last val step:     {val_points[-1][0]}")
            print(f"  Last val loss:     {val_points[-1][1]:.4f}")

        if success_points:
            print(f"  Last success step: {success_points[-1][0]}")
            print(f"  Last success:      {success_points[-1][1]:.1f}%")


if __name__ == "__main__":
    main()
