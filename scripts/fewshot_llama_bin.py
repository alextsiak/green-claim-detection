from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import csv
import random

mode = "original"  # can be changed to "exorde"

dataset_train = load_dataset("alextsiak/green-claims-twitter-balanced", split="train")
    
if mode == "original":
    dataset_test = load_dataset("alextsiak/green-claims-twitter-balanced", split="test")
elif mode == "exorde":
    dataset_test = load_dataset("alextsiak/green_claims_annotated", split="train")

if mode == "original":
    label_col = "label_binary"
    text_col = "tweet"
elif mode == "exorde":
    label_col = "classification"
    text_col = "text"

label_names = dataset_train.features[label_col].names

model_name = "meta-llama/Llama-3.1-8B-Instruct"

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

chosen_k = 1  #selected based on dev set macro-F1
few_shot_ds = few_shot_sets[chosen_k]

with open("../prompts/prompt_bin_fewshot.txt") as f:
    instructions = f.read()

def build_few_shot_prompt(few_shot_ds, query_text, label_names, text_col="tweet", label_column="label_multi"):
    example_blocks = "\n".join(
        f"Post: {ex[text_col]}\nLabel: {label_names[ex[label_column]]}"
        for ex in few_shot_ds
    )
    query_block = f"Now classify this post:\nPost: {query_text}\nLabel:"
    return f"{instructions}\n{example_blocks}\n\n{query_block}"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

results = []
for example in dataset_test:
    if mode == "original":
        tweet_text = example[text_col]
        gold_label = label_names[example[label_col]]
    elif mode == "exorde":
        tweet_text = example[text_col]
        gold_label = example[label_col]
    prompt = build_few_shot_prompt(few_shot_ds, tweet_text, label_names, text_col=text_col, label_column=label_col)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    results.append({
        "tweet": tweet_text,
        "gold_label": gold_label,
        "prediction": response.strip()
    })

with open(f"../results/predictions_llama_bin_fewshot_{mode}_k{chosen_k}.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["tweet", "gold_label", "prediction"])
    writer.writeheader()
    writer.writerows(results)

print(f"Done: test set predictions saved for k={chosen_k}")