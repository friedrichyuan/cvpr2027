#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
third_party_dir="${THIRD_PARTY_DIR:-${repo_root}/third_party}"

egoinfinity_repo="${EGOINFINITY_REPO_URL:-https://github.com/Rice-RobotPI-Lab/EgoInfinity.git}"
egoinfinity_ref="${EGOINFINITY_REF:-de59610531c08010d63d0493d6a72973eabd402e}"

do_as_i_do_repo="${DO_AS_I_DO_REPO_URL:-https://github.com/malik-group/do-as-i-do.git}"
do_as_i_do_ref="${DO_AS_I_DO_REF:-b5a617060a970ba2d9af7b2e216903c643cda185}"

spider_repo="${SPIDER_REPO_URL:-https://github.com/facebookresearch/spider.git}"
spider_ref="${SPIDER_REF:-2e54f19ee6ab8e0690c0f585beb1e9f8f53a6898}"

dinov2_repo="${DINOV2_REPO_URL:-https://github.com/facebookresearch/dinov2.git}"
dinov2_ref="${DINOV2_REF:-7764ea0f912e53c92e82eb78a2a1631e92725fc8}"

geocalib_repo="${GEOCALIB_REPO_URL:-https://github.com/cvg/GeoCalib.git}"
geocalib_ref="${GEOCALIB_REF:-97b8968e7798a66bf04fcf791fb535624241bda7}"

submodules="${SETUP_THIRD_PARTY_SUBMODULES:-1}"
force="${SETUP_THIRD_PARTY_FORCE:-0}"

mkdir -p "${third_party_dir}"

checkout_repo() {
  local name="$1"
  local url="$2"
  local ref="$3"
  local dest="${third_party_dir}/${name}"

  if [[ -e "${dest}" && ! -d "${dest}/.git" ]]; then
    if [[ "${force}" != "1" ]]; then
      echo "Refusing to overwrite non-git path: ${dest}" >&2
      echo "Set SETUP_THIRD_PARTY_FORCE=1 to remove and recreate it." >&2
      exit 1
    fi
    rm -rf "${dest}"
  fi

  if [[ ! -d "${dest}/.git" ]]; then
    echo "[third_party] clone ${name} from ${url}"
    git clone "${url}" "${dest}"
  fi

  echo "[third_party] checkout ${name}@${ref}"
  git -C "${dest}" fetch --tags origin
  if git -C "${dest}" rev-parse --verify --quiet "${ref}^{commit}" >/dev/null; then
    git -C "${dest}" checkout --detach "${ref}"
  else
    git -C "${dest}" fetch origin "${ref}"
    git -C "${dest}" checkout --detach FETCH_HEAD
  fi

  if [[ "${submodules}" == "1" && -f "${dest}/.gitmodules" ]]; then
    echo "[third_party] update submodules for ${name}"
    git -C "${dest}" submodule update --init --recursive
  fi
}

checkout_repo "EgoInfinity" "${egoinfinity_repo}" "${egoinfinity_ref}"
checkout_repo "do-as-i-do" "${do_as_i_do_repo}" "${do_as_i_do_ref}"
checkout_repo "SPIDER" "${spider_repo}" "${spider_ref}"
checkout_repo "dinov2" "${dinov2_repo}" "${dinov2_ref}"
checkout_repo "GeoCalib" "${geocalib_repo}" "${geocalib_ref}"

verify_nested_repo() {
  local nested="$1"
  local expected="$2"
  local nested_path="${third_party_dir}/${nested}"
  local actual
  if ! git -C "${nested_path}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Missing required Do-as-I-Do reconstruction module: ${nested_path}" >&2
    echo "Verify that recursive submodule checkout succeeded." >&2
    exit 1
  fi
  actual="$(git -C "${nested_path}" rev-parse HEAD)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Unexpected commit for ${nested}: ${actual}; expected ${expected}" >&2
    exit 1
  fi
}

verify_nested_repo \
  "do-as-i-do/reconstruction/modules/sam3" \
  "b8e18f5f021fb27c4f6eeb1a8f4ecfa3fa0bec39"
verify_nested_repo \
  "do-as-i-do/reconstruction/modules/sam-3d-objects" \
  "875b010016f4c86936985b74ffcd7230e6979d02"
verify_nested_repo \
  "do-as-i-do/reconstruction/modules/Fast-SAM3D" \
  "823d4784dbf73a9a58ca8b2384dee1eaf8fc1c6b"

lock_file="${third_party_dir}/THIRD_PARTY_LOCK.tsv"
{
  printf "name\turl\trequested_ref\tresolved_commit\n"
  for name in EgoInfinity do-as-i-do SPIDER dinov2 GeoCalib; do
    case "${name}" in
      EgoInfinity) url="${egoinfinity_repo}"; ref="${egoinfinity_ref}" ;;
      do-as-i-do) url="${do_as_i_do_repo}"; ref="${do_as_i_do_ref}" ;;
      SPIDER) url="${spider_repo}"; ref="${spider_ref}" ;;
      dinov2) url="${dinov2_repo}"; ref="${dinov2_ref}" ;;
      GeoCalib) url="${geocalib_repo}"; ref="${geocalib_ref}" ;;
    esac
    commit="$(git -C "${third_party_dir}/${name}" rev-parse HEAD)"
    printf "%s\t%s\t%s\t%s\n" "${name}" "${url}" "${ref}" "${commit}"
  done
  for nested in \
    do-as-i-do/reconstruction/modules/sam3 \
    do-as-i-do/reconstruction/modules/sam-3d-objects \
    do-as-i-do/reconstruction/modules/Fast-SAM3D
  do
    url="$(git -C "${third_party_dir}/${nested}" remote get-url origin)"
    commit="$(git -C "${third_party_dir}/${nested}" rev-parse HEAD)"
    printf "%s\t%s\t%s\t%s\n" "${nested}" "${url}" "${commit}" "${commit}"
  done
} > "${lock_file}"

cat <<EOF

Third-party source trees are ready under:
  ${third_party_dir}

Resolved commits were recorded in:
  ${lock_file}

These files are intentionally ignored by Git. Keep model weights, MANO files,
AoE data, Hugging Face caches, and experiment outputs outside this repository
or under ignored local paths.

Common next steps:
  conda activate egoinfinity
  pip install -e "${third_party_dir}/EgoInfinity"

  conda activate dai-retarget
  pip install -e "${third_party_dir}/do-as-i-do/retargeting"

  conda activate spider
  pip install -e "${third_party_dir}/SPIDER"

The defaults are the commits validated by this release. Override the *_REF
variables only when deliberately validating a different upstream revision.
EOF
