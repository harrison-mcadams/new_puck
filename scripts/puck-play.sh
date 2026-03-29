#!/bin/bash

# Simple wrapper for the Kodi stream launcher
# Usage: ./puck-play.sh --url "https://..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_EXEC="${SCRIPT_DIR}/../.venv/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    # Fallback if venv is missing
    PYTHON_EXEC="python3"
fi

$PYTHON_EXEC "${SCRIPT_DIR}/kodi_play.py" "$@"
