#!/bin/bash

# Navigate to the directory where the script is
cd "$(dirname "$0")"

# Activate your Python environment if needed (optional)
# source path/to/venv/bin/activate

# Run the Streamlit app
python -m streamlit run webapp.py
