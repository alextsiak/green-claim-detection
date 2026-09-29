import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from datasets import ClassLabel
from transformers import DataCollatorWithPadding, Trainer, EvalPrediction, EarlyStoppingCallback

model_name = "climatebert/distilroberta-base-climate-f"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(text):
    text = text["tweet"]
    #mentions replaced with special token
    text = re.sub(r'@\w+', '[USER]', text)
    #replace URLs with special token
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
    #lowercase
    text = text.lower()
    tokenized_text = tokenizer(text, truncation=True)
    return tokenized_text


data_train = load_dataset("alextsiak/green-claims-twitter-balanced", split="train")
data_dev = load_dataset("alextsiak/green-claims-twitter-balanced", split="validation")
data_test = load_dataset("alextsiak/green-claims-twitter-balanced", split="test")

cols_to_remove = ['Unnamed: 0', 'id', 'tweet', 'username', 'domain','label_multi']

tokenized_train = data_train.map(preprocess, remove_columns=cols_to_remove)
tokenized_dev = data_dev.map(preprocess, remove_columns=cols_to_remove)
tokenized_test = data_test.map(preprocess, remove_columns=cols_to_remove)

tokenized_train = tokenized_train.rename_column("label_binary", "labels")
tokenized_dev = tokenized_dev.rename_column("label_binary", "labels")
tokenized_test = tokenized_test.rename_column("label_binary", "labels")

data_collator = DataCollatorWithPadding(tokenizer)

label2id = {"not_green": 0, "green_claim": 1}
id2label = {0: "not_green", 1: "green_claim"}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,
                                                           id2label=id2label,
                                                           label2id=label2id)
training_args = TrainingArguments(
    output_dir="/climatebert-bin",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    fp16=True
)


def metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro_average = f1_score(y_true=labels, y_pred=predictions, average='macro')
    metrics = {'f1_macro': f1_macro_average,
               'accuracy': acc}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    predictions = preds.argmax(axis=-1)
    labels = p.label_ids
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(y_true=labels, y_pred=predictions, average='macro')
    return {'f1_macro': f1_macro, 'accuracy': acc}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

results = trainer.evaluate(tokenized_test)
print(results)
trainer.push_to_hub()