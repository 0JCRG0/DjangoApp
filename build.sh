#!/usr/bin/env bash
# exit on error
set -o errexit

# create and activate the virtual environment
python3 -m venv env1
source env1/bin/activate

# install dependencies
pip install -r requirements.txt

# collect static files and run migrations
python3 manage.py collectstatic --no-input
python3 manage.py migrate