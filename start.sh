#!/bin/bash
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

# Activate the virtual environment
if [ -d ".venv" ]; then
    echo "✦ Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "✦ Virtual environment (.venv) not found. Setting up using 'uv'..."
    if command -v uv &> /dev/null; then
        uv sync
        source .venv/bin/activate
    else
        echo "❌ Error: 'uv' is not installed. Please install 'uv' or create the virtual environment manually."
        exit 1
    fi
fi

# Start the server
echo "🚀 Starting ReImagineX server..."
python src/server.py
