from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import csv

dataset_test = load_dataset("alextsiak/green_claims_annotated", split="train")
model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

with open("../prompts/prompt_span_detection.txt") as f:
    prompt_template = f.read()

results = []
    
for example in dataset_test:
    tweet_text = example["text"]
    gold_label = example["classification"]  
    gold_span = example["label"]
    prompt = prompt_template.format(tweet=tweet_text)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(response.strip())
    results.append({
        "tweet": tweet_text,
        "gold_label": gold_label,
        "gold_span": gold_span,
        "prediction": response
    })


with open(f"../results/predictions_llama_span.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["tweet", "gold_label", "gold_span", "prediction"])
    writer.writeheader()
    writer.writerows(results)