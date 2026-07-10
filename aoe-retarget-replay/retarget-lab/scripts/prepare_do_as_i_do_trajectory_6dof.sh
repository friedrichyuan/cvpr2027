#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${RETARGETING_PYTHON:-python}"
run_name="foundation_jar_visual_demo_v1"
task="foundation_jar_bimanual_leftscale"
raw_dir=""
mode="symlink"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) run_name="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --raw-dir) raw_dir="$2"; shift 2 ;;
    --copy) mode="copy"; shift ;;
    -h|--help)
      echo "Usage: $0 --raw-dir PATH [--run-name NAME] [--task NAME] [--copy]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$raw_dir" ]]; then
  echo "Missing --raw-dir" >&2
  exit 2
fi
if [[ ! -d "$raw_dir" ]]; then
  echo "raw-dir does not exist: $raw_dir" >&2
  exit 2
fi

exp="$repo_root/experiments/$run_name"
dst="$exp/intermediates/trajectory_6dof/do_as_i_do/reconstruction/raw_dir"
mkdir -p "$(dirname "$dst")" "$exp/logs"
rm -rf "$dst"
if [[ "$mode" == "copy" ]]; then
  mkdir -p "$dst"
  rsync -a --delete "$raw_dir"/ "$dst"/
else
  ln -s "$raw_dir" "$dst"
fi

"$python_bin" "$repo_root/scripts/extract_do_as_i_do_trajectory_6dof.py" \
  --raw-dir "$dst" \
  --experiment-root "$exp" \
  --task "$task" \
  --mode symlink | tee "$exp/logs/trajectory_6dof_do_as_i_do.log"
