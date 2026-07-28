#!/usr/bin/env bash
set -euo pipefail

readonly DAI_PINNED_COMMIT="b5a617060a970ba2d9af7b2e216903c643cda185"
readonly DAI_OFFICIAL_REMOTE="https://github.com/malik-group/do-as-i-do.git"

usage() {
  cat <<'EOF'
Run the pinned, unmodified Do-as-I-Do pipeline.

Required:
  --raw-dir PATH                 Prepared Do-as-I-Do input directory.
  --task NAME                    Native Do-as-I-Do task basename.
  --output-root-dir PATH         New archive path for native outputs.

Runtime options:
  --cuda-visible-devices VALUE   Value exposed as CUDA_VISIBLE_DEVICES.
  --egl-device-id ID             Logical EGL device (requires --headless).
  --headless                     Pass native --no-show-viewer and use EGL.
  --no-wait                      Pass native --no-wait-on-finish.
  --check-only                   Validate paths, runtime, and minimum input files.

Environment:
  RETARGETING_PYTHON             Python executable for the backend.
  DAI_PRISTINE_SOURCE            Clean checkout, bundle, or Git URL.
  DAI_PRISTINE_RUNTIME           Cache for the pinned clean checkout.
  DAI_PRISTINE_REMOTE            Clone URL when no local checkout is available.
  DAI_CUDA_VISIBLE_DEVICES       Alias for --cuda-visible-devices.
  DAI_EGL_DEVICE_ID              Alias for --egl-device-id.

The pinned launch.py does not consistently propagate --output-root-dir. This
wrapper therefore runs it unchanged in a disposable checkout, lets it use its
native retargeting/outputs directory, and moves that directory to the requested
archive after exit. A backend return code of zero plus a decodable robot video
is a candidate result; final acceptance is by manual video review.
EOF
}

die() {
  echo "run_do_as_i_do_official_retarget: $*" >&2
  exit 2
}

require_value() {
  local option="$1"
  local remaining="$2"
  (( remaining >= 2 )) || die "$option requires a value"
}

raw_dir=""
task=""
output_root_dir=""
cuda_visible_devices="${DAI_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
egl_device_id="${DAI_EGL_DEVICE_ID:-${EGL_DEVICE_ID:-}}"
headless=0
no_wait=0
check_only=0

while (( $# > 0 )); do
  case "$1" in
    --raw-dir)
      require_value "$1" "$#"
      raw_dir="$2"
      shift 2
      ;;
    --task)
      require_value "$1" "$#"
      task="$2"
      shift 2
      ;;
    --output-root-dir)
      require_value "$1" "$#"
      output_root_dir="$2"
      shift 2
      ;;
    --cuda-visible-devices)
      require_value "$1" "$#"
      cuda_visible_devices="$2"
      shift 2
      ;;
    --egl-device-id)
      require_value "$1" "$#"
      egl_device_id="$2"
      shift 2
      ;;
    --headless)
      headless=1
      shift
      ;;
    --no-wait)
      no_wait=1
      shift
      ;;
    --check-only)
      check_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "$raw_dir" ]] || die "missing --raw-dir"
[[ -n "$task" ]] || die "missing --task"
[[ -n "$output_root_dir" ]] || die "missing --output-root-dir"
if (( headless == 0 )) && [[ -n "$egl_device_id" ]]; then
  die "--egl-device-id requires --headless"
fi
if (( headless == 1 )) && [[ -z "$egl_device_id" ]]; then
  egl_device_id="0"
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
manifest_python="$(command -v python3 || true)"
[[ -n "$manifest_python" ]] || die "python3 is required"
python_bin="${RETARGETING_PYTHON:-$(command -v python3 || true)}"
[[ -n "$python_bin" && -x "$python_bin" ]] || \
  die "RETARGETING_PYTHON is not executable: $python_bin"

if ! "$manifest_python" - "$task" <<'PY'
import re
import sys

task = sys.argv[1]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", task) or task in {".", ".."}:
    raise SystemExit(1)
PY
then
  die "--task must be a safe basename"
fi

raw_dir="$("$manifest_python" - "$raw_dir" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().resolve(strict=True)
if not path.is_dir():
    raise SystemExit(1)
print(path)
PY
)" || die "cannot resolve --raw-dir"

output_root_dir="$("$manifest_python" - "$output_root_dir" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
if os.path.lexists(path):
    raise SystemExit(1)
print(path.resolve(strict=False))
PY
)" || die "--output-root-dir must be a new path"

if ! "$manifest_python" - "$raw_dir" "$output_root_dir" <<'PY'
import os
import sys
from pathlib import Path

raw = Path(sys.argv[1])
output = Path(sys.argv[2])
common = Path(os.path.commonpath((str(raw), str(output))))
if common in {raw, output}:
    raise SystemExit(1)
PY
then
  die "raw-dir and output-root-dir must be disjoint"
fi

[[ -f "$raw_dir/config.json" ]] || die "missing input config.json"
[[ -f "$raw_dir/$task/all_hand_meshes.npz" ]] || \
  die "missing input $task/all_hand_meshes.npz"

output_parent="$(dirname "$output_root_dir")"
mkdir -p "$output_parent"
output_parent="$(cd "$output_parent" && pwd -P)"
output_root_dir="$output_parent/$(basename "$output_root_dir")"

default_local_source="$repo_root/third_party/do-as-i-do"
if git -C "$default_local_source" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  pristine_source="${DAI_PRISTINE_SOURCE:-$default_local_source}"
else
  pristine_source="${DAI_PRISTINE_SOURCE:-${DAI_PRISTINE_REMOTE:-$DAI_OFFICIAL_REMOTE}}"
fi
pristine_runtime="${DAI_PRISTINE_RUNTIME:-${XDG_CACHE_HOME:-$HOME/.cache}/aoe-retarget/pristine-do-as-i-do-$DAI_PINNED_COMMIT}"
pristine_runtime="$("$manifest_python" - "$pristine_runtime" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
)"

verify_checkout() {
  local checkout="$1"
  local label="$2"
  local head
  git -C "$checkout" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
    die "$label is not a Git checkout: $checkout"
  head="$(git -C "$checkout" rev-parse --verify HEAD)" || \
    die "cannot read $label commit"
  [[ "$head" == "$DAI_PINNED_COMMIT" ]] || \
    die "$label commit is $head, expected $DAI_PINNED_COMMIT"
  [[ -z "$(git -C "$checkout" status --porcelain=v1 --untracked-files=all)" ]] || \
    die "$label is dirty: $checkout"
}

if git -C "$pristine_source" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  verify_checkout "$pristine_source" "DAI_PRISTINE_SOURCE"
fi

if [[ ! -e "$pristine_runtime" ]]; then
  runtime_parent="$(dirname "$pristine_runtime")"
  mkdir -p "$runtime_parent"
  staging_runtime="$(mktemp -d "$runtime_parent/.pristine-do-as-i-do.XXXXXXXX")"
  git clone --quiet --no-checkout --no-hardlinks "$pristine_source" "$staging_runtime" || \
    die "cannot clone DAI source"
  git -C "$staging_runtime" checkout --quiet --detach "$DAI_PINNED_COMMIT" || \
    die "DAI source does not contain pinned commit"
  verify_checkout "$staging_runtime" "new DAI runtime"
  mv "$staging_runtime" "$pristine_runtime"
fi
verify_checkout "$pristine_runtime" "DAI_PRISTINE_RUNTIME"

if (( check_only == 1 )); then
  printf 'DAI_CHECK_ONLY_OK commit=%s task=%s raw_dir=%s\n' \
    "$DAI_PINNED_COMMIT" "$task" "$raw_dir"
  exit 0
fi

run_workspace="$(mktemp -d "$output_parent/.aoe-dai-run.XXXXXXXX")"
disposable_runtime="$run_workspace/backend"
keep_workspace=0
cleanup_workspace() {
  if [[ -d "${run_workspace:-}" && "$keep_workspace" -eq 0 ]]; then
    rm -rf "$run_workspace"
  fi
}
trap cleanup_workspace EXIT

git clone --quiet --no-checkout --no-hardlinks \
  "$pristine_runtime" "$disposable_runtime" || \
  die "cannot create disposable DAI checkout"
git -C "$disposable_runtime" checkout --quiet --detach "$DAI_PINNED_COMMIT" || \
  die "cannot checkout pinned DAI commit"
verify_checkout "$disposable_runtime" "disposable DAI checkout"

native_output_root="$disposable_runtime/retargeting/outputs"
if [[ -e "$native_output_root" ]]; then
  git -C "$disposable_runtime" sparse-checkout init --no-cone || \
    die "cannot initialize output isolation"
  git -C "$disposable_runtime" sparse-checkout set \
    '/*' '!/retargeting/outputs/' || die "cannot isolate native output directory"
fi
[[ ! -e "$native_output_root" ]] || die "native output directory could not be isolated"

backend_python_dir="$(cd "$(dirname "$python_bin")" && pwd -P)"
backend_env=(
  "HOME=${HOME:-/tmp}"
  "USER=${USER:-unknown}"
  "LOGNAME=${LOGNAME:-${USER:-unknown}}"
  "PATH=$backend_python_dir:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  "LANG=C.UTF-8"
  "LC_ALL=C.UTF-8"
  "PYTHONUNBUFFERED=1"
  "PYTHONDONTWRITEBYTECODE=1"
)
if [[ -n "$cuda_visible_devices" ]]; then
  backend_env+=("CUDA_VISIBLE_DEVICES=$cuda_visible_devices")
fi
if (( headless == 1 )); then
  backend_env+=(
    "MUJOCO_GL=egl"
    "PYOPENGL_PLATFORM=egl"
    "EGL_DEVICE_ID=$egl_device_id"
  )
elif [[ -n "${DISPLAY:-}" ]]; then
  backend_env+=("DISPLAY=$DISPLAY")
fi

backend_argv=("launch.py" "--raw-dir" "$raw_dir" "--task" "$task")
(( headless == 0 )) || backend_argv+=("--no-show-viewer")
(( no_wait == 0 )) || backend_argv+=("--no-wait-on-finish")

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "DAI pinned commit: $DAI_PINNED_COMMIT"
echo "DAI raw input: $raw_dir"
echo "DAI archive destination: $output_root_dir"
printf 'DAI native argv:'
printf ' %q' "$python_bin" "${backend_argv[@]}"
printf '\n'

previous_directory="$PWD"
cd "$disposable_runtime/retargeting"
keep_workspace=1
set +e
env -i "${backend_env[@]}" "$python_bin" "${backend_argv[@]}"
backend_rc=$?
set -e
cd "$previous_directory"
ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "$native_output_root"
if ! mv "$native_output_root" "$output_root_dir"; then
  echo "native output publication failed; workspace preserved: $run_workspace" >&2
  exit 74
fi

manifest_path="$output_root_dir/aoe_dai_run_manifest.json"
"$manifest_python" - \
  "$manifest_path" "$backend_rc" "$DAI_PINNED_COMMIT" "$raw_dir" "$task" \
  "$output_root_dir" "$python_bin" "$started_at" "$ended_at" \
  "${backend_argv[@]}" <<'PY'
import json
import sys
from pathlib import Path

(
    manifest,
    backend_rc,
    commit,
    raw_dir,
    task,
    output_root,
    python_bin,
    started_at,
    ended_at,
    *argv,
) = sys.argv[1:]

payload = {
    "schema_version": 1,
    "policy": "backend_success_and_manual_video_review",
    "manual_review_required": True,
    "backend": {
        "name": "do-as-i-do",
        "commit": commit,
        "returncode": int(backend_rc),
        "source_modified": False,
    },
    "input": {"raw_dir": raw_dir, "task": task},
    "output_root_dir": output_root,
    "python": python_bin,
    "argv": argv,
    "started_at_utc": started_at,
    "ended_at_utc": ended_at,
}
Path(manifest).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

keep_workspace=0
exit "$backend_rc"
