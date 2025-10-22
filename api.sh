#!/bin/bash
set -e

# ---- Initialize pyenv and pyenv-virtualenv ----
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# ---- Activate the virtualenv ----
echo "Activating pyenv virtualenv senet..."
pyenv activate senet

# ---- Install requirements ----
if [ -f requirements.txt ] && [ -s requirements.txt ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt is empty or missing. Installing torch and torchvision..."
    pip install torch torchvision
fi

# ---- Run main.py ----
echo "Running main.py..."
python3 main.py
