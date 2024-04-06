#!/bin/bash
if ! command -v pipreqs -h&> /dev/null
then
    echo "pipreqs is not installed. Install pipreq to automate python dependencies"
    exit 1
else
    echo "pipreqs checked"
fi

rm -fr .venv

pipreqs --force
# Check if venv module is available
if ! command -v python3 -m venv &> /dev/null
then
    echo "Python venv module is not available. Please install it."
    exit 1
fi

# Create a virtual environment
python3 -m venv .venv || { echo "Failed to create the virtual environment"; exit 1; }
echo "Virtual environement created in .venv"
# Activate the virtual environment
source .venv/bin/activate || { echo "Failed to activate the virtual environment"; exit 1; }
echo "Virtual environemnt activated"
# Install packages from requirements.txt
pip install -r requirements.txt || { echo "Failed to install the required packages"; exit 1; }
echo "Requirements successfully installed"
if ! grep -qxF '.venv' .gitignore; then
  echo '.venv' >> .gitignore
fi
# Deactivate the virtual environment
#deactivate