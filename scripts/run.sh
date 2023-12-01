export OPENAI_API_KEY=[OPENAI_API_KEY]
export OPENAI_ORG=[OPENAI_ORG]

python run.py \
    --generate_config_path configs/example_config.json \
    --overwrite True \
    --budget 20