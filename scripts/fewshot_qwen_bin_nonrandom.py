from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import csv

mode = "original"  # can be changed to "exorde"

if mode == "original":
    dataset_test = load_dataset("alextsiak/green-claims-twitter-balanced", split="test")
elif mode == "exorde":
    dataset_test = load_dataset("alextsiak/green_claims_annotated", split="train")

model_name = "Qwen/Qwen3.5-4B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

with open("../prompts/prompt_bin_fewshot_k3.txt") as f:
    prompt_template = f.read()

results = []
    
for example in dataset_test:
    if mode == "original":
        tweet_text = example["tweet"]
        gold_label = example["label_binary"]
    elif mode == "exorde":
        tweet_text = example["text"]
        gold_label = example["classification"]
    prompt = prompt_template.format(tweet=tweet_text)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=50)
    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(response.strip())
    results.append({
        "tweet": tweet_text,
        "gold_label": gold_label,
        "prediction": response
    })


with open(f"../results/predictions_qwen_bin_fewshot_nonrandom_k3_{mode}.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["tweet", "gold_label", "prediction"])
    writer.writeheader()
    writer.writerows(results)