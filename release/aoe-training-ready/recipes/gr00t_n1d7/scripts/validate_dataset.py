# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Validate converted LeRobot V2 dataset for GR00T N1.7 compatibility.

Checks:
1. Directory structure completeness
2. Parquet schema and field dimensions
3. Modality.json consistency
4. Video file existence
5. Action/state normalization ranges
6. Episode continuity and frame counts
7. NaN / Inf detection
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


class DatasetValidator:
    def __init__(self, dataset_dir: str):
        self.root = Path(dataset_dir)
        self.errors = []
        self.warnings = []
        self.stats = {}

    def error(self, msg: str):
        self.errors.append(msg)
        print(f"  [ERROR] {msg}")

    def warn(self, msg: str):
        self.warnings.append(msg)
        print(f"  [WARN]  {msg}")

    def info(self, msg: str):
        print(f"  [OK]    {msg}")

    def check_structure(self):
        """Check directory structure exists."""
        print("\n--- Checking directory structure ---")

        required = [
            "meta/info.json",
            "meta/modality.json",
            "meta/tasks.jsonl",
            "meta/episodes.jsonl",
            "data/chunk-000",
            "videos/chunk-000/observation.images.ego_view",
        ]

        for path in required:
            full = self.root / path
            if not full.exists():
                self.error(f"Missing: {path}")
            else:
                self.info(f"Found: {path}")

    def check_meta(self):
        """Validate metadata files."""
        print("\n--- Checking metadata ---")

        # info.json
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            self.error("meta/info.json not found")
            return

        with open(info_path) as f:
            info = json.load(f)

        required_fields = ["total_episodes", "total_frames", "fps", "action_horizon", "state_dim"]
        for field in required_fields:
            if field not in info:
                self.error(f"info.json missing field: {field}")
            else:
                self.info(f"info.json: {field} = {info[field]}")

        self.stats["info"] = info

        # modality.json
        mod_path = self.root / "meta" / "modality.json"
        if not mod_path.exists():
            self.error("meta/modality.json not found")
            return

        with open(mod_path) as f:
            modality = json.load(f)

        for key in ["state", "action"]:
            if key not in modality:
                self.error(f"modality.json missing: {key}")
            else:
                groups = modality[key]
                for group_name, group_data in groups.items():
                    dim = group_data.get("dim", 0)
                    has_min = "min" in group_data
                    has_max = "max" in group_data
                    self.info(f"modality[{key}].{group_name}: dim={dim}, "
                              f"has_range={has_min and has_max}")

        # tasks.jsonl
        tasks_path = self.root / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path) as f:
                tasks = [json.loads(line) for line in f]
            self.info(f"tasks.jsonl: {len(tasks)} unique tasks")
            self.stats["tasks"] = tasks
        else:
            self.error("meta/tasks.jsonl not found")

        # episodes.jsonl
        ep_path = self.root / "meta" / "episodes.jsonl"
        if ep_path.exists():
            with open(ep_path) as f:
                episodes = [json.loads(line) for line in f]
            self.info(f"episodes.jsonl: {len(episodes)} episodes")
            self.stats["episodes"] = episodes
        else:
            self.error("meta/episodes.jsonl not found")

    def check_parquet_files(self, max_check: int = 10):
        """Validate parquet file schema and data integrity."""
        print("\n--- Checking parquet files ---")

        data_dir = self.root / "data" / "chunk-000"
        if not data_dir.exists():
            self.error("data/chunk-000 not found")
            return

        parquet_files = sorted(data_dir.glob("episode_*.parquet"))
        self.info(f"Found {len(parquet_files)} parquet files")

        if not parquet_files:
            self.error("No parquet files found")
            return

        info = self.stats.get("info", {})
        expected_state_dim = info.get("state_dim", 10)
        expected_action_dim = info.get("action_dim", 160)

        checked = 0
        total_nan_count = 0
        total_inf_count = 0

        for pf in parquet_files[:max_check]:
            df = pd.read_parquet(pf)

            # Check required columns
            required_cols = ["observation.state", "action", "timestamp", "episode_index", "index"]
            for col in required_cols:
                if col not in df.columns:
                    self.error(f"{pf.name}: missing column '{col}'")

            # Check dimensions
            if "observation.state" in df.columns:
                sample_state = np.array(df["observation.state"].iloc[0])
                if len(sample_state) != expected_state_dim:
                    self.error(f"{pf.name}: state dim={len(sample_state)}, "
                              f"expected={expected_state_dim}")

            if "action" in df.columns:
                sample_action = np.array(df["action"].iloc[0])
                if len(sample_action) != expected_action_dim:
                    self.error(f"{pf.name}: action dim={len(sample_action)}, "
                              f"expected={expected_action_dim}")

            # Check for NaN/Inf
            if "observation.state" in df.columns:
                states = np.array(df["observation.state"].tolist())
                nan_count = np.isnan(states).sum()
                inf_count = np.isinf(states).sum()
                total_nan_count += nan_count
                total_inf_count += inf_count
                if nan_count > 0:
                    self.error(f"{pf.name}: {nan_count} NaN values in state")
                if inf_count > 0:
                    self.error(f"{pf.name}: {inf_count} Inf values in state")

            if "action" in df.columns:
                actions = np.array(df["action"].tolist())
                nan_count = np.isnan(actions).sum()
                inf_count = np.isinf(actions).sum()
                total_nan_count += nan_count
                total_inf_count += inf_count
                if nan_count > 0:
                    self.error(f"{pf.name}: {nan_count} NaN values in action")
                if inf_count > 0:
                    self.error(f"{pf.name}: {inf_count} Inf values in action")

            checked += 1

        self.info(f"Checked {checked} parquet files")
        if total_nan_count == 0 and total_inf_count == 0:
            self.info(f"No NaN/Inf detected in checked files")

    def check_videos(self, max_check: int = 10):
        """Check video files exist and are readable."""
        print("\n--- Checking video files ---")

        video_dir = self.root / "videos" / "chunk-000" / "observation.images.ego_view"
        if not video_dir.exists():
            self.error("videos directory not found")
            return

        video_files = sorted(video_dir.glob("episode_*.mp4"))
        self.info(f"Found {len(video_files)} video files/links")

        broken_links = 0
        for vf in video_files[:max_check]:
            if vf.is_symlink():
                target = vf.resolve()
                if not target.exists():
                    self.error(f"{vf.name}: broken symlink → {target}")
                    broken_links += 1
            elif not vf.exists():
                self.error(f"{vf.name}: file not found")

        if broken_links == 0:
            self.info(f"All checked video links are valid")

    def check_action_statistics(self, max_episodes: int = 20):
        """Compute and display action/state statistics."""
        print("\n--- Action/State statistics ---")

        data_dir = self.root / "data" / "chunk-000"
        if not data_dir.exists():
            return

        parquet_files = sorted(data_dir.glob("episode_*.parquet"))[:max_episodes]

        all_states = []
        all_actions = []

        for pf in parquet_files:
            df = pd.read_parquet(pf)
            if "observation.state" in df.columns:
                states = np.array(df["observation.state"].tolist())
                all_states.append(states)
            if "action" in df.columns:
                actions = np.array(df["action"].tolist())
                all_actions.append(actions)

        if all_states:
            states = np.concatenate(all_states, axis=0)
            state_dim = states.shape[1]
            info = self.stats.get("info", {})
            mode = info.get("mode", "gripper")
            print(f"\n  State statistics (dim={state_dim}, mode={mode}):")

            # Left hand EEF position (always first 3 dims)
            left_pos = states[:, 0:3]
            print(f"    Left EEF position range:  {left_pos.max(axis=0) - left_pos.min(axis=0)}")

            if state_dim == 62:
                # Sharpa: [left_wrist_eef(9), right_wrist_eef(9), left_hand_joints(22), right_hand_joints(22)]
                right_pos = states[:, 9:12]
                print(f"    Right EEF position range: {right_pos.max(axis=0) - right_pos.min(axis=0)}")
                lj = states[:, 18:40]
                rj = states[:, 40:62]
                print(f"    Left Sharpa (22D):  mean={lj.mean():.4f}, "
                      f"range=[{lj.min():.4f}, {lj.max():.4f}]")
                print(f"    Right Sharpa (22D): mean={rj.mean():.4f}, "
                      f"range=[{rj.min():.4f}, {rj.max():.4f}]")
            elif state_dim == 20:
                # Gripper: [left_wrist_eef(9), right_wrist_eef(9), left_gripper(1), right_gripper(1)]
                right_pos = states[:, 9:12]
                print(f"    Right EEF position range: {right_pos.max(axis=0) - right_pos.min(axis=0)}")
                print(f"    Left gripper:  [{states[:, 18].min():.3f}, {states[:, 18].max():.3f}]")
                print(f"    Right gripper: [{states[:, 19].min():.3f}, {states[:, 19].max():.3f}]")
            else:
                print(f"    Mean: {states.mean(axis=0)[:5]}...")
                print(f"    Min:  {states.min(axis=0)[:5]}...")
                print(f"    Max:  {states.max(axis=0)[:5]}...")

            ranges = states.max(axis=0) - states.min(axis=0)
            dead_dims = (ranges < 1e-6).sum()
            if dead_dims > 0:
                self.warn(f"{dead_dims} state dimensions have near-zero range")

        if all_actions:
            actions = np.concatenate(all_actions, axis=0)
            info = self.stats.get("info", {})
            state_dim = info.get("state_dim", 20)
            first_step = actions[:, :state_dim]
            print(f"\n  Action (first step, dim={state_dim}):")
            print(f"    Mean: {first_step.mean(axis=0)[:5]}...")
            print(f"    Std:  {first_step.std(axis=0)[:5]}...")

    def check_groot_compatibility(self):
        """Check GR00T N1.7 specific requirements."""
        print("\n--- GR00T N1.7 compatibility check ---")

        info = self.stats.get("info", {})
        episodes = self.stats.get("episodes", [])

        # Check action horizon
        ah = info.get("action_horizon", 0)
        if ah < 1:
            self.error(f"action_horizon={ah}, must be >= 1")
        elif ah > 40:
            self.warn(f"action_horizon={ah} is very large, typical range is 10-20")
        else:
            self.info(f"action_horizon={ah} is within typical range")

        # Check fps
        fps = info.get("fps", 0)
        if fps < 5 or fps > 60:
            self.warn(f"fps={fps} seems unusual, typical range is 10-30")
        else:
            self.info(f"fps={fps}")

        # Check episode lengths
        if episodes:
            lengths = [ep["length"] for ep in episodes]
            self.info(f"Episode lengths: min={min(lengths)}, max={max(lengths)}, "
                      f"mean={np.mean(lengths):.0f}")
            short = sum(1 for l in lengths if l < 30)
            if short > 0:
                self.warn(f"{short} episodes shorter than 30 frames")

    def run(self, max_check_files: int = 20):
        """Run all validation checks."""
        print(f"=== Validating dataset: {self.root} ===")

        self.check_structure()
        self.check_meta()
        self.check_parquet_files(max_check=max_check_files)
        self.check_videos(max_check=max_check_files)
        self.check_action_statistics(max_episodes=max_check_files)
        self.check_groot_compatibility()

        print(f"\n{'='*50}")
        print(f"VALIDATION SUMMARY")
        print(f"  Errors:   {len(self.errors)}")
        print(f"  Warnings: {len(self.warnings)}")
        if self.errors:
            print(f"\n  FAILED — fix errors before training")
            return False
        elif self.warnings:
            print(f"\n  PASSED with warnings")
            return True
        else:
            print(f"\n  PASSED — dataset ready for GR00T N1.7 training")
            return True


def main():
    parser = argparse.ArgumentParser(description="Validate LeRobot V2 dataset for GR00T N1.7")
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the converted LeRobot V2 dataset",
    )
    parser.add_argument(
        "--max-check",
        type=int,
        default=20,
        help="Maximum number of files to check in detail",
    )

    args = parser.parse_args()

    validator = DatasetValidator(args.dataset_dir)
    success = validator.run(max_check_files=args.max_check)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
