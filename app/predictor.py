"""Loads the trained model once and scores essays.

Same logic as the notebook (Steps 5, 8, 9, 11 and 15). The prompt texts are read
from fast_heads.joblib, so the website uses exactly the prompt the model was trained on.
"""
from unsloth import FastLanguageModel  # import unsloth before torch
import json
import re

import joblib
import numpy as np
import torch

LORA_DIR = "models/llama3_aes_lora"
HEADS_FILE = "models/fast_heads.joblib"
MAX_SEQ_LEN = 1280
THRESHOLD = 0.2

# 1. Load the fast classifiers and the prompt texts saved by the notebook
saved = joblib.load(HEADS_FILE)
scaler, heads, RANGES = saved["scaler"], saved["heads"], saved["ranges"]
TRAITS = list(heads)

# 2. Load LLaMA-3-8B with our LoRA weights
model, tokenizer = FastLanguageModel.from_pretrained(
    LORA_DIR, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True)
FastLanguageModel.for_inference(model)


def build_prompt(essay):
    essay = " ".join(essay.split()[:700])
    fmt = json.dumps({t: 0 for t in TRAITS})
    return saved["template"].format(fmt=fmt, rubric=saved["rubric"],
                                    prompt=saved["essay_prompt"], essay=essay)


def parse_scores(text):
    match = re.search(r"\{.*?\}", text, re.S)
    try:
        scores = json.loads(match.group())
        return {t: int(np.clip(round(float(scores[t])), *RANGES[t])) for t in TRAITS}
    except Exception:
        return None


@torch.no_grad()
def slow_score(essay):
    inputs = tokenizer(build_prompt(essay), return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return parse_scores(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))


@torch.no_grad()
def embed(essay):
    inputs = tokenizer(build_prompt(essay), return_tensors="pt").to("cuda")
    out = model(**inputs, output_hidden_states=True)
    if out.hidden_states is not None:
        hidden = out.hidden_states[-1]
    else:
        hidden = model.get_base_model().model(**inputs).last_hidden_state
    return hidden[0].mean(dim=0).float().cpu().numpy()


def score_essay(essay):
    """Fast score + confidence for every trait; slow module for low-confidence traits."""
    x = scaler.transform(embed(essay)[None, :])
    result = {}
    for t, head in heads.items():
        probs = head.predict_proba(x)[0]
        k = len(probs)
        result[t] = {
            "score": int(head.classes_[probs.argmax()]),
            "max": RANGES[t][1],
            "confidence": round(float((probs.max() - 1 / k) / (1 - 1 / k)), 2),
            "module": "fast",
        }

    low = [t for t in TRAITS if result[t]["confidence"] < THRESHOLD]
    if low:
        slow = slow_score(essay)
        if slow:
            for t in low:
                result[t].update(score=slow[t], module="slow")
    return result
