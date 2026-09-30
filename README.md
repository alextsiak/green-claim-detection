# Detecting Green Claims in Social Media Posts Using Large Language Models

This repository hosts the code for detecting environmental/green claims in social media posts using zero-shot, few-shot, and fine-tuned language models.

## Structure

`prompts/`: prompts for the binary and multi-class green claim detection (zero-shot and few-shot), as well as for the span detection

`scripts/`: scripts for fine-tuning ClimateBERT and Qwen-3.5-4B, tuning the k value of the few-shot examples, running inference (zero-shot, few-shot), as well as span detection for Qwen-3.5-4B and Llama-3.1-8B-Instruct.


## Installation

```bash
git clone https://github.com/alextsiak/green-claim-detection.git
cd green-claim-detection

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

All script can be run by: `python scripts/script_name.py`

To run the Qwen fine-tuning:
```
python scripts/finetune_qwen3_green_claims.py \
    --dataset_name ... \
    --label_mode binary (or multi) \
    --output_dir ./outputs/qwen3-4b-green-binary
```
