#!/bin/sh

python -m autopager train autopager/data autopager/autopager.crf
./setup.py sdist upload
./setup.py bdist_wheel upload
