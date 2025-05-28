#!/bin/bash
python3 -m venv wor_ai_env
source wor_ai_env/bin/activate
pip install -r requirements.txt
echo "Setup complete. Run with: source wor_ai_env/bin/activate && python main.py"