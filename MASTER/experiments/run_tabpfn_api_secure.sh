#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv/bin/python. Create/install the project virtualenv first." >&2
  exit 1
fi

printf "Prior Labs API key: "
IFS= read -r -s TABPFN_TOKEN
printf "\n"

if [[ -z "${TABPFN_TOKEN}" ]]; then
  echo "Empty API key; stopping." >&2
  exit 1
fi

export TABPFN_TOKEN
trap 'unset TABPFN_TOKEN' EXIT

.venv/bin/python scripts/master_tabpfn_api_experiments.py --allow-api-upload "$@"
