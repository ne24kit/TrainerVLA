# Reproduction Scripts

Run from the project root.

1. Set up Python packages and local LeRobot patches:

```bash
./scripts/repro/setup_env.sh
```

2. Download local model assets and Robometer code:

```bash
./scripts/repro/download_assets.sh
```

3. If `lerobot-edit-dataset split` fails on the known AV1 file, repair only that video:

```bash
./scripts/repro/fix_libero_goal_video.sh
```

4. Rebuild replay datasets for Task 2:

```bash
./scripts/repro/prepare_task2_replay_datasets.sh
```

Set `FORCE=1` only when you intentionally want to rebuild existing replay datasets.

5. Launch experiment groups:

```bash
./scripts/repro/run_task1_baseline.sh
./scripts/repro/run_task2_replay.sh
./scripts/repro/run_task2_replan_eval.sh
./scripts/repro/run_task2_lora.sh
./scripts/repro/run_task4_reward_free.sh
```

By default `run_task1_baseline.sh` assumes the seen checkpoint already exists at
`outputs/smolvla_libero90_pretrain/checkpoints/070000/pretrained_model`.
To reproduce the full seen pretrain too:

```bash
RUN_SEEN_PRETRAIN=1 ./scripts/repro/run_task1_baseline.sh
```
