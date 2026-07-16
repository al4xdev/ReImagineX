#!/usr/bin/env bash
set -euo pipefail
# ReImagineX Startup Script
# This script activates the virtual environment and starts the FastAPI server.

# Locate the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Initialize .env if not existing
if [ ! -f .env ]; then
    echo "✦ Creating .env file from template..."
    cp .env.example .env
    echo "✔ .env file created. Please update it with your settings."
fi

if ! command -v uv &> /dev/null; then
    echo "Error: uv is required. Install it from https://docs.astral.sh/uv/."
    exit 1
fi

# Start the server
echo "🚀 Starting ReImagineX server..."
exec uv run python -m src.server
