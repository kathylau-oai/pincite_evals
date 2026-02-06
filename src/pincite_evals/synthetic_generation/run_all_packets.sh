#!/usr/bin/env bash
set -euo pipefail

# Run one end-to-end synthetic generation pass for every packet directory.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
packet_root="${repo_root}/data/case_law_packets"

# Optional first argument lets you pin run ids for reproducibility.
run_timestamp="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! -d "${packet_root}" ]]; then
  echo "Packet root not found: ${packet_root}" >&2
  exit 1
fi

temp_config_dir="$(mktemp -d /tmp/pincite-synth-all-packets-XXXX)"
trap 'rm -rf "${temp_config_dir}"' EXIT

for packet_dir in "${packet_root}"/*; do
  [[ -d "${packet_dir}" ]] || continue

  packet_id="$(basename "${packet_dir}")"
  run_id="${run_timestamp}_${packet_id}"
  config_path="${temp_config_dir}/${packet_id}.yaml"

  cat > "${config_path}" <<EOF
packet_id: "${packet_id}"
EOF

  echo "Running packet_id=${packet_id} run_id=${run_id}"
  (
    cd "${repo_root}"
    uv run python -m pincite_evals.synthetic_generation.cli run-all \
      --config "${config_path}" \
      --run-id "${run_id}"
  )
done

