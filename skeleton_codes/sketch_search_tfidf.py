"""
Fast TF-IDF search over questions.json (no embeddings).

Usage:
  uv run skeleton_codes/sketch_search_tfidf.py --query "your question"
  uv run skeleton_codes/sketch_search_tfidf.py --bench 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def load_questions(path: str | Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "question_id": int(d["question_id"]),
            "question_example": str(d["question_example"]),
        }
        for d in data
        if "question_id" in d and "question_example" in d
    ]


def generate_test_queries(items: List[Dict], n: int = 10):
    if not items:
        return []
    n = max(1, min(n, len(items)))
    idxs = np.linspace(0, len(items) - 1, num=n, dtype=int)

    syn_map = {
        "export": "save",
        "import": "load",
        "create": "build",
        "delete": "remove",
        "update": "modify",
        "list": "enumerate",
        "error": "exception",
        "query": "search",
        "api": "application programming interface",
        "db": "database",
        "database": "db",
        "file": "document",
        "folder": "directory",
        "path": "filepath",
        "image": "picture",
        "csv": "comma separated values",
        "json": "javascript object notation",
        "sql": "structured query language",
        "url": "link",
    }
    stop = set(
        "a an the of in on at to for from with by and or is are be how what".split()
    )

    def syn_replace(s: str) -> str:
        t = s.lower()
        for k, v in syn_map.items():
            t = re.sub(rf"\b{re.escape(k)}\b", v, t)
        return t

    def reorder(s: str) -> str:
        t = s.strip().rstrip("?!.")
        m = re.search(r"\b(in|with|for)\b\s+(.+)", t, flags=re.IGNORECASE)
        if m:
            pre = t[: m.start()].strip(",;:. ")
            tail = m.group(2).strip()
            return f"In {tail}, {pre}?" if pre else t
        return f"Please advise: {t}?"

    def noise(s: str, other: str) -> str:
        ws = [w for w in re.findall(r"[a-zA-Z0-9]+", other.lower()) if len(w) > 3]
        random.shuffle(ws)
        extra = " ".join(ws[:3])
        base = s.strip().rstrip("?!. ")
        return f"{base} {extra}?"

    def drop_stop(s: str) -> str:
        ws = [w for w in re.findall(r"[a-zA-Z0-9]+", s.lower()) if w not in stop]
        return " ".join(ws)

    tests = []
    for idx in idxs:
        text = items[idx]["question_example"].strip()
        qid = items[idx]["question_id"]
        other = items[(idx + 1) % len(items)]["question_example"]
        variants = [
            text,
            syn_replace(text),
            reorder(text),
            noise(text, other),
            drop_stop(text),
            f"Could you explain {text.rstrip('?').lower()}?",
        ]
        for q in variants:
            tests.append((q, qid))
    return tests


def main():
    ap = argparse.ArgumentParser(description="TF-IDF search over questions.json")
    ap.add_argument("--questions-file", default="questions.json")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--query", default=None)
    ap.add_argument("--bench", type=int, default=0)
    args = ap.parse_args()

    items = load_questions(args.questions_file)
    ids = [x["question_id"] for x in items]
    texts = [x["question_example"] for x in items]

    t0 = time.perf_counter()
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), norm="l2")
    X = vect.fit_transform(texts)
    build_s = time.perf_counter() - t0

    def search(q: str):
        t = time.perf_counter()
        Xq = vect.transform([q])
        sim = (Xq @ X.T).toarray().ravel()
        order = np.argsort(-sim)[: args.top_k]
        latency_ms = round((time.perf_counter() - t) * 1000, 2)
        res = [
            {
                "rank": i + 1,
                "question_id": int(ids[idx]),
                "text": texts[idx],
                "score": float(sim[idx]),
            }
            for i, idx in enumerate(order)
        ]
        margin = float(sim[order[0]] - sim[order[1]]) if len(order) > 1 else 0.0
        return {"top": res, "latency_ms": latency_ms, "margin": margin}

    if args.query:
        out = search(args.query)
        print(
            json.dumps(
                {
                    "build_s": round(build_s, 3),
                    "query": args.query,
                    "latency_ms": out["latency_ms"],
                    "margin": round(out["margin"], 4),
                    "top": out["top"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    bench_n = args.bench or 10
    tests = generate_test_queries(items, bench_n)
    correct = 0
    lat = []
    for q, expected_id in tests:
        out = search(q)
        lat.append(out["latency_ms"])
        top1 = out["top"][0]
        correct += int(top1["question_id"] == expected_id)
        print(
            json.dumps(
                {"q": q, "expected_id": expected_id, "top1": top1}, ensure_ascii=False
            )
        )
    print(
        json.dumps(
            {
                "build_s": round(build_s, 3),
                "tests": len(tests),
                "avg_latency_ms": round(float(np.mean(lat)), 2) if lat else 0.0,
                "top1_accuracy": round(correct / len(tests), 3) if tests else 0.0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
