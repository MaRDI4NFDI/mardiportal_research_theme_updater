#!/usr/bin/env bash
# Run extract_formulas.py with lakeFS credentials from environment variables.
#
# Usage:
#   ./maintenance/extract_formulas_run.sh              # list available papers
#   ./maintenance/extract_formulas_run.sh Q6190920    # extract formulas for one paper

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Help / env-var check
# ---------------------------------------------------------------------------
missing=()
[[ -z "${LAKEFS_USER:-}"     ]] && missing+=("LAKEFS_USER")
[[ -z "${LAKEFS_PASSWORD:-}" ]] && missing+=("LAKEFS_PASSWORD")

if [[ ${#missing[@]} -gt 0 ]] || [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    cat <<EOF
Usage: $(basename "$0") [QID]

  Without QID : list all papers available in lakeFS and exit.
  With QID    : extract formulas from that paper and write
                maintenance/{QID}_formulas__{model}.json.

Required environment variables:
  LAKEFS_USER        lakeFS access key ID
  LAKEFS_PASSWORD    lakeFS secret access key

Backend selection (only required when a QID is given):
  OLLAMA_MODEL set             -> self-hosted Ollama/LiteLLM backend
                                   (requires OLLAMA_URL, OLLAMA_API_TOKEN)
  OLLAMA_MODEL unset, OPENROUTER_MODEL set -> OpenRouter backend
                                   (requires OPENROUTER_API_KEY; prompted
                                   interactively if not set)
  Neither set                  -> error, no implicit default backend

Optional environment variables (defaults shown):
  LAKEFS_URL          https://lake-bioinfmed.zib.de
  LAKEFS_REPO         mardi-fdo-data
  LAKEFS_BRANCH       main
  SPARQL_ENDPOINT_URL https://query.portal.mardi4nfdi.de/sparql

  OpenRouter backend:
    OPENROUTER_MODEL             (no default; e.g. z-ai/glm-4.7-flash)
    OPENROUTER_MAX_TOKENS        131072
    OPENROUTER_PROVIDER          (unset; comma-separated provider slugs, e.g. "cloudflare,novita")
    OPENROUTER_REASONING_TOKENS  8000

  Ollama/LiteLLM backend:
    OLLAMA_URL                   (no default; e.g. https://tllm.science-berlin.de)
    OLLAMA_API_TOKEN             (no default)
    OLLAMA_API_STYLE              openai (set to "native" for a plain Ollama/Open WebUI
                                  deployment with no LiteLLM-compatible /model/info —
                                  confirmed needed for https://ollama.zib.de)
    OLLAMA_NUM_CTX                32768 (only used when OLLAMA_API_STYLE=native)
    OLLAMA_DISABLE_THINKING      true
    OLLAMA_REASONING_EFFORT      (unset)

Examples:
  LAKEFS_USER=<access-key-id> \\
  LAKEFS_PASSWORD=<secret-access-key> \\
  OPENROUTER_API_KEY=<openrouter-key> \\
  OPENROUTER_MODEL=z-ai/glm-4.7-flash \\
  $(basename "$0") Q6190920

  LAKEFS_USER=<access-key-id> \\
  LAKEFS_PASSWORD=<secret-access-key> \\
  OLLAMA_URL=https://tllm.science-berlin.de \\
  OLLAMA_API_TOKEN=<token> \\
  OLLAMA_MODEL=zib/gpt-oss-20b \\
  $(basename "$0") Q6190920

  LAKEFS_USER=<access-key-id> \\
  LAKEFS_PASSWORD=<secret-access-key> \\
  OLLAMA_URL=https://ollama.zib.de/ollama/api \\
  OLLAMA_API_TOKEN=<token> \\
  OLLAMA_MODEL=qwen3.5:35b \\
  OLLAMA_API_STYLE=native \\
  OLLAMA_NUM_CTX=100000 \\
  $(basename "$0") Q6190920
EOF
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo ""
        echo "Error: missing required environment variable(s): ${missing[*]}" >&2
        exit 1
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# OpenRouter API key — only needed in extract mode (QID given) and only when
# the Ollama backend isn't selected (OLLAMA_MODEL takes precedence, matching
# resolve_route()'s backend precedence in extract_formulas.py).
# ---------------------------------------------------------------------------
if [[ -n "${1:-}" ]] && [[ -z "${OLLAMA_MODEL:-}" ]] && [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    read -rsp "OpenRouter API key: " OPENROUTER_API_KEY
    echo
    export OPENROUTER_API_KEY
fi

# ---------------------------------------------------------------------------
# Defaults for optional vars
# ---------------------------------------------------------------------------
export LAKEFS_URL="${LAKEFS_URL:-https://lake-bioinfmed.zib.de}"
export LAKEFS_REPO="${LAKEFS_REPO:-mardi-fdo-data}"
export LAKEFS_BRANCH="${LAKEFS_BRANCH:-main}"
export SPARQL_ENDPOINT_URL="${SPARQL_ENDPOINT_URL:-https://query.portal.mardi4nfdi.de/sparql}"

exec python maintenance/extract_formulas.py "$@"
