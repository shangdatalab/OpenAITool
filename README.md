# OpenAI Generator

## Goal
This project aims at providing a tool for dataset-level OpenAI prompting. The key features are
1. It allows saving outputs every several inference, results can be picked up from last termination.
2. It takes customized prepare_data and post_process functions.
3. There is a consecutive prompting that takes a list of prompts. (Sometimes you might want to breakdown the prompting into several substeps.)

## Install
```bash
python -m pip install -r requirements.txt
```

## Example Usage
```bash
bash scripts/run.sh
```
Please see arguments in `openai_tools.py`