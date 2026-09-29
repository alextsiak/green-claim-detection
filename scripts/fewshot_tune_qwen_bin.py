from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import csv
import random

dataset_train = load_dataset("alextsiak/green-claims-twitter-balanced", split="train")
dataset_val = load_dataset("alextsiak/green-claims-twitter-balanced", split="validation")

label_col = "label_binary"
label_names = dataset_train.features[label_col].names 

model_name = "Qwen/Qwen3-4B"

def get_few_shot_samples(train_data, label_column, max_k=3, seed=42):
    rng = random.Random(seed)
    labels = train_data.unique(label_column)
    indices_by_class = {}
    for label in labels:
        class_indices = [i for i, ex in enumerate(train_data) if ex[label_column] == label]
        rng.shuffle(class_indices)
        indices_by_class[label] = class_indices

    nested_samples = {}
    for k in range(1, max_k + 1):
        selected_indices = []
        for label, idx_list in indices_by_class.items():
            selected_indices.extend(idx_list[:k])
        nested_samples[k] = train_data.select(selected_indices)
    return nested_samples

few_shot_sets = get_few_shot_samples(dataset_train, label_column=label_col, max_k=3, seed=42)

for k, ds in few_shot_sets.items():
    for ex in ds:
        print(ex["tweet"], "->", label_names[ex[label_col]])
    ds.to_csv(f"qwen_few_shot_examples_k{k}.csv")

    
with open("../prompts/prompt_bin_fewshot.txt") as f:
    instructions = f.read()

def build_few_shot_prompt(few_shot_ds, query_text, label_names, text_col="tweet", label_column="label_binary"):
    example_blocks = "\n".join(
        f"Post: {ex[text_col]}\nLabel: {label_names[ex[label_column]]}"
        for ex in few_shot_ds
    )
    query_block = f"Now classify this post:\nPost: {query_text}\nLabel:"
    return f"{instructions}\n{example_blocks}\n{query_block}"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

#run generation for each k, on the dev set
for k, few_shot_ds in few_shot_sets.items():
    results = []
    for example in dataset_val:
        tweet_text = example["tweet"]
        gold_label = label_names[example[label_col]]
        prompt = build_few_shot_prompt(few_shot_ds, tweet_text, label_names, label_column=label_col)

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=10)
        generated_ids = output[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        results.append({
            "tweet": tweet_text,
            "gold_label": gold_label,
            "prediction": response.strip()
        })

    with open(f"/results/predictions_qwen_bin_fewshot_k{k}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tweet", "gold_label", "prediction"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Done: k={k}, {len(results)} predictions saved to predictions_qwen_bin_fewshot_k{k}.csv")