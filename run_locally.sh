#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

ENV_FILE=".env"

usage() {
    cat <<'EOF'
Usage: ./run_locally.sh [--dry-run] [--themes-only] [--] [extra topic_overviews args...]

Requires a local .env file in this directory. The script exports variables from
.env, validates the required settings, and runs:

    python -m topic_overviews

Examples:
    ./run_locally.sh --dry-run
    ./run_locally.sh --themes-only
    ./run_locally.sh

For harvest runs, TOPIC_OVERVIEWS_MODEL_QID must point to a KG model item with
P1966 "LLM model identifier"; that value is passed to the selected LLM provider.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE is required for local runs." >&2
    echo "Create it from .env.example and fill in the local credentials/settings." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

dry_run=false
themes_only=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=true ;;
        --themes-only) themes_only=true ;;
    esac
done

missing=()
require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        missing+=("$name")
    fi
}

require_var "TOPIC_OVERVIEWS_RESEARCH_THEME_QID"
require_var "SPARQL_ENDPOINT_URL"

if [[ "${TOPIC_OVERVIEWS_RESEARCH_THEME_QID:-}" == "Q0" ]]; then
    echo "ERROR: TOPIC_OVERVIEWS_RESEARCH_THEME_QID must be set to the real research-theme class QID." >&2
    exit 1
fi

if [[ "$dry_run" == false ]]; then
    require_var "MEDIAWIKI_API_URL"
    require_var "MEDIAWIKI_BOT_USER"
    require_var "MEDIAWIKI_BOT_PASSWORD"
    require_var "WIKIBASE_URL"
fi

llm_provider="${TOPIC_OVERVIEWS_LLM_PROVIDER:-anthropic}"
llm_provider="${llm_provider,,}"
if [[ "$llm_provider" != "anthropic" && "$llm_provider" != "openai" ]]; then
    echo "ERROR: TOPIC_OVERVIEWS_LLM_PROVIDER must be 'anthropic' or 'openai'." >&2
    exit 1
fi

if [[ "$themes_only" == false ]]; then
    require_var "TOPIC_OVERVIEWS_MODEL_QID"
fi

if [[ "$themes_only" == false && "$llm_provider" == "anthropic" ]]; then
    require_var "ANTHROPIC_API_KEY"
fi

if [[ "$themes_only" == false && "$llm_provider" == "openai" ]]; then
    require_var "TOPIC_OVERVIEWS_OPENAI_BASE_URL"
    require_var "TOPIC_OVERVIEWS_OPENAI_API_KEY"
fi

if (( ${#missing[@]} > 0 )); then
    echo "ERROR: Missing required environment variables in $ENV_FILE:" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    exit 1
fi

python -m topic_overviews "$@"
