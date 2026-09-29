from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from datasets import load_dataset
import csv

mode = "original" #can be changed to exorde

if mode == "original":
    dataset_test = load_dataset("alextsiak/green-claims-twitter-balanced", split="test")
elif mode == "exorde":
    dataset_test = load_dataset("alextsiak/green_claims_annotated", split="train")

model_name = "meta-llama/Llama-3.1-8B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(load_in_4bit=False)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=bnb_config
)

with open("../prompts/prompt_multi.txt") as f:
    prompt_template = f.read()

results = []
    
for example in dataset_test:
    if mode == "original":
        tweet_text = example["tweet"]
        gold_label = example["label_multi"]
    elif mode == "exorde":
        tweet_text = example["text"]
        gold_label = example["classification"]
    prompt = prompt_template.format(tweet=tweet_text)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(response.strip())
    results.append({
        "tweet": tweet_text,
        "gold_label": gold_label,
        "prediction": response
    })


if mode == "original":
    result_file = "../results/predictions_llama_multi_zero.csv"
elif mode == "exorde":
    result_file = "../results/predictions_llama_multi_zero_exorde.csv"

with open(result_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["tweet", "gold_label", "prediction"])
    writer.writeheader()
    writer.writerows(results)