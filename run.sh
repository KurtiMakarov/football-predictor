#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

if [[ -z "${API_FOOTBALL_KEY:-}" ]]; then
  echo "API_FOOTBALL_KEY is not set."
  read -r -s -p "Enter API_FOOTBALL_KEY (input hidden): " API_FOOTBALL_KEY
  echo
  export API_FOOTBALL_KEY
fi

export FLASK_APP=src.web
export FLASK_ENV=development
export FLASK_DEBUG="${FLASK_DEBUG:-1}"
export PORT="${PORT:-5000}"

python3 -m flask run --host=0.0.0.0 --port="${PORT}"
