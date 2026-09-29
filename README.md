# Detecting Green Claims in Social Media Posts Using Large Language Models

This repository hosts the code for running different experiments for green claim detection in social media posts. 

## Structure

`prompts/`: prompts for the binary and multi-class green claim detection (zero-shot and few-shot), as well as for the span detection

`scripts/`: scripts for fine-tuning ClimateBERT and Qwen-3.5-4B, tuning the k value of the few-shot examples, running inference (zero-shot, few-shot), as well as span detection for Qwen-3.5-4B and Llama-3.1-8B-Instruct.
