#!/usr/bin/env bash
# Register / update the topic-overviews Prefect deployment.
# Pulls the API auth string from the production K8s cluster and runs
# workflow_deploy_prefect.py.
set -euo pipefail

cd "$(dirname "$0")"

PREFECT_API_URL="http://prefect-mardi.zib.de/api"
PREFECT_API_AUTH_STRING=$(kubectl get secret prefect-server-auth -n production \
    -o jsonpath='{.data.auth-string}' | base64 -d)

export PREFECT_API_URL
export PREFECT_API_AUTH_STRING

python3.10 workflow_deploy_prefect.py
