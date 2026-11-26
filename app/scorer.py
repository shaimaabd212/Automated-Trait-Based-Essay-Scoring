"""Loads the fine-tuned LLaMA-3-8B and scores an essay (same logic as notebook Step 15)."""
import json
import re

import joblib
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "unsloth/llama-3-8b-bnb-4bit"
LORA_DIR = "models/llama3_aes_lora"
HEADS_FILE = "models/fast_heads.joblib"
THRESHOLD = 0.2

saved = joblib.load(HEADS_FILE)
scaler, heads, RANGES = saved["scaler"], saved["heads"], saved["ranges"]
TRAITS = list(heads.keys())


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map={"": 0}, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model.eval()
    return model, tokenizer


def build_prompt(essay):
    essay = " ".join(essay.split()[:700])
    fmt = json.dumps({t: 0 for t in TRAITS})
    return saved["template"].format(
        fmt=fmt, rubric=saved["rubric"], prompt=saved["essay_prompt"], essay=essay
    )


@torch.no_grad()
def embed(essay, model, tokenizer):
    inputs = tokenizer(build_prompt(essay), return_tensors="pt").to("cuda")
    out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1][0].mean(dim=0).float().cpu().numpy()


def parse_scores(text):
    match = re.search(r"\{.*?\}", text, re.S)
    try:
        scores = json.loads(match.group())
        return {t: int(np.clip(round(float(scores[t])), *RANGES[t])) for t in TRAITS}
    except Exception:
        return None


@torch.no_grad()
def slow_score(essay, model, tokenizer):
    inputs = tokenizer(build_prompt(essay), return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return parse_scores(text)


def fast_score(x):
    xs = scaler.transform(x)
    scores, conf = {}, {}
    for t, head in heads.items():
        probs = head.predict_proba(xs)
        k = probs.shape[1]
        scores[t] = head.classes_[probs.argmax(1)]
        conf[t] = (probs.max(1) - 1 / k) / (1 - 1 / k)
    return pd.DataFrame(scores), pd.DataFrame(conf)


def score_essay(essay, model, tokenizer):
    scores, conf = fast_score(embed(essay, model, tokenizer)[None, :])
    result = {
        t: {"score": int(scores[t][0]), "confidence": round(float(conf[t][0]), 2), "module": "fast"}
        for t in TRAITS
    }
    low = [t for t in TRAITS if conf[t][0] < THRESHOLD]
    if low:
        slow = slow_score(essay, model, tokenizer)
        if slow:
            for t in low:
                result[t].update(score=slow[t], module="slow")
    return result
