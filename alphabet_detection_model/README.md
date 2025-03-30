## Overview
This folder contains the Static Gesture Classification Model codebase. It is capable of predicting alphanumeric gestures.

## Launch the Static Gesture Classification model
Make sure to source the virtual environment first. It can be found in the base folder ../
The requirements.txt can be used to create a virtual environment.

```bash
python -m venv ../venv
source ../venv/bin/activate
pip install -r ../requirements.txt # For Linux
pip install -r ../requirements_mac.txt # For Mac
```
```bash
python run_live.py --model "../combined/model/static_model.h5" --dataset "../combined/data/mp_data_static" 
```