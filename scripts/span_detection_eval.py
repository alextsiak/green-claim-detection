import json
import csv
import re
import pandas as pd
import difflib

df = pd.read_csv("../results/predictions_qwen_span_exorde.csv")

#label normalization (for gold labels)

labels_norm = {'implicit green claim': 'Implicit_Green', 'explicit green claim': 'Explicit_Green'}

def safe_parse(x):
    if pd.isna(x):
        return []
    return json.loads(x)

df['gold_parsed'] = df['gold_span'].apply(safe_parse)

for i in range(len(df)):
    for gold_span in df['gold_parsed'][i]:
        span_text = gold_span['text']
        start = gold_span['start']
        end = gold_span['end']
        label = gold_span['labels'][0]   
        label = labels_norm.get(label, label)
        gold_span['labels'][0] = label

print(df)

#finding character offsets for predicted spans

def find_span_offsets(source, pred, fuzzy_threshold=0.8):
    if not pred:
        return None,None,'empty_pred'
    
    #exact match
    idx = source.find(pred)
    if idx != -1:
        return idx, idx + len(pred), 'exact'

    #case-insensitive match
    idx = source.lower().find(pred.lower())
    if idx != -1:
        return idx, idx + len(pred), 'case_insensitive'

    #fuzzy
    #window of pred's length slides across source and best-scoring alignment is kept
    best_ratio = 0
    best_start = 0
    window = len(pred)
    for i in range(len(source) - window + 1):
        candidate = source[i:i+window]
        ratio = difflib.SequenceMatcher(None, candidate, pred).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
    if best_start is not None and best_ratio >= fuzzy_threshold:
        return best_start, best_start + window, f'fuzzy_{best_ratio:.2f}'
    return None, None, 'unresolved'



df['pred_parsed'] = df['prediction'].apply(json.loads)

for i in range(len(df)):
    source = df['tweet'][i]
    for pred_span in df['pred_parsed'][i]:
        start, end, match_type = find_span_offsets(source, pred_span['span'])
        pred_span['start'] = start
        pred_span['end'] = end
        pred_span['match_type'] = match_type

#print(df['pred_parsed'][2])

#df.to_csv("test.csv")



def compute_precision(row,mode="hard"):
    gold_spans = row['gold_parsed']
    pred_spans = row['pred_parsed']
    total = 0
    overlap_ratio = 0


    #no predicted spans
    if len(pred_spans) == 0:
        return None
    
    for pred_span in pred_spans:
        pred_type = pred_span['type']
        pred_start = pred_span['start']
        pred_end = pred_span['end']

        #couldn't locate pred text in source
        if pred_start is None or pred_end is None:
            total += 0
            continue

        #find best-overlapping gold span for this prediction
        best_overlap_ratio = 0
        for gold_span in gold_spans:
            gold_type = gold_span['labels'][0]
            gold_start = gold_span['start']
            gold_end = gold_span['end']

            overlap = max(0, min(pred_end, gold_end) - max(pred_start, gold_start))
            if mode == "hard" and gold_type != pred_type:
                overlap_ratio = 0
            else:
                overlap_ratio = overlap / (pred_end - pred_start)
            
            best_overlap_ratio = max(best_overlap_ratio, overlap_ratio)

        total += best_overlap_ratio
    return total / len(pred_spans)


def compute_recall(row, mode="hard"):
    gold_spans = row['gold_parsed']
    pred_spans = row['pred_parsed']

    #no gold spans
    if len(gold_spans) == 0:
        return None

    total = 0
    overlap_ratio = 0
    for gold_span in gold_spans:
        gold_type = gold_span['labels'][0]
        gold_start = gold_span['start']
        gold_end = gold_span['end']

        #find best-overlapping predicted span for this gold span
        best_overlap_ratio = 0
        for pred_span in pred_spans:
            pred_type = pred_span['type']
            pred_start = pred_span['start']
            pred_end = pred_span['end']

            if pred_start is None or pred_end is None:
                continue

            overlap = max(0, min(pred_end, gold_end) - max(pred_start, gold_start))

            if mode == "hard" and gold_type != pred_type:
                overlap_ratio = 0
            else:
                overlap_ratio = overlap / (gold_end - gold_start)

            best_overlap_ratio = max(best_overlap_ratio, overlap_ratio)

        total += best_overlap_ratio

    return total / len(gold_spans)


df['precision_hard'] = df.apply(lambda row: compute_precision(row, mode="hard"), axis=1)
df['recall_hard'] = df.apply(lambda row: compute_recall(row, mode="hard"), axis=1)

df['precision_soft'] = df.apply(lambda row: compute_precision(row, mode="soft"), axis=1)
df['recall_soft'] = df.apply(lambda row: compute_recall(row, mode="soft"), axis=1)

precision_hard = df['precision_hard'].mean(skipna=True)
recall_hard = df['recall_hard'].mean(skipna=True)
f1_hard = 2 * precision_hard * recall_hard / (precision_hard + recall_hard)

precision_soft = df['precision_soft'].mean(skipna=True)
recall_soft = df['recall_soft'].mean(skipna=True)   
f1_soft = 2 * precision_soft * recall_soft / (precision_soft + recall_soft)

f1_delta = f1_soft - f1_hard

print(f1_hard, f1_soft, f1_delta)
