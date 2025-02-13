# Fine Tuning LLM Models in your Mac with mlx lm

## Steps:

- activate your conda environment `conda activate <env_name>` (my python version is 3.12.8)
- install necessary packages `pip install -r requirements.txt`
- login to huggingface `huggingface-cli login`
- run `huggingface-cli download unsloth/Llama-3.2-1B-Instruct --local-dir ./Llama-3.2-1B-Instruct` to download the model
- run `inference.py` to test the model
- run `fine_tuning.py` to fine tune the model

## Notes:
- after fine tuning, the adapter will be saved in `./Llama-3.2-1B-Instruct-adapters` directory
- to use the fine tuned model, just pass the `apapter_path` to the `inference.py` script:

```python
model, tokenizer = load(
    "./Llama-3.2-1B-Instruct",
    adapter_path="./Llama-3.2-1B-Instruct-adapters",
)
```

## Important links:
https://github.com/ml-explore/mlx-examples.git
https://heidloff.net/article/apple-mlx-fine-tuning/
https://medium.com/@levchevajoana/fine-tuning-llms-with-lora-and-mlx-lm-c0b143642deb
