#!/bin/bash
rm -fr .venv

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
echo "Virtual environemtn activated"
# Install packages from requirements.txt
pip install -r requirements.txt || { echo "Failed to install the required packages"; exit 1; }
echo "Requirements successfully installed"
# Deactivate the virtual environment
#deactivate