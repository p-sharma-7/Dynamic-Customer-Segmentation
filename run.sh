#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for the Streamlit dashboard
# Usage: bash run.sh

if [ -f .venv/bin/activate ]; then
	source .venv/bin/activate
elif [ -f venv/bin/activate ]; then
	source venv/bin/activate
fi

echo "Installing required packages (skip if already satisfied)..."
pip install -q -r requirements.txt

echo "Starting Streamlit app..."
exec streamlit run app.py --server.runOnSave true

