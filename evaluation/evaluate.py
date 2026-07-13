import json
from pathlib import Path
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from models import verifier
from models.retriever import BM25Retriever

CLAIMS_PATH = Path("data/claims_dev.jsonl")
CORPUS_PATH = Path("data/corpus.jsonl")
CHECKPOINT_PATH = Path("models/saved/verifier_unweighted")
OUT_PATH = Path("evaluation/contrast_eval_results.json")

K = 5  

LABEL2ID = {"CONTRADICT": 0, "SUPPORT": 1, "NOT_ENOUGH_INFO": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def verify_claim_against_doc(
    verifier, claim: str, doc: Dict,
    threshold_nei: float = 0.5,
) -> Dict:

    sentences = doc.get("abstract", [])
    if not sentences:
        return {"label": "NOT_ENOUGH_INFO", "confidence": 0.0, "evidence_sentence": None}

    pairs = [(sent, claim) for sent in sentences]
    probs = verifier.predict_batch(pairs)

    relevance = np.maximum(probs[:, 0], probs[:, 1])
    best_idx = int(relevance.argmax())
    best_probs = probs[best_idx]

    support_score, contradict_score, nei_score = best_probs[1], best_probs[0], best_probs[2]
    max_sr = max(support_score, contradict_score)

    if max_sr < threshold_nei:
        label, confidence = "NOT_ENOUGH_INFO", float(nei_score)
    elif support_score >= contradict_score:
        label, confidence = "SUPPORT", float(support_score)
    else:
        label, confidence = "CONTRADICT", float(contradict_score)

    return {"label": label, "confidence": confidence, "evidence_sentence": sentences[best_idx]}


def aggregate_across_docs(doc_results: List[Dict]) -> Dict:
    non_nei = [r for r in doc_results if r["label"] != "NOT_ENOUGH_INFO"]
    if non_nei:
        return max(non_nei, key=lambda r: r["confidence"])
    if doc_results:
        return max(doc_results, key=lambda r: r["confidence"])
    return {"label": "NOT_ENOUGH_INFO", "confidence": 0.0, "evidence_sentence": None}


def resolve_gold_label(claim_obj: Dict) -> str:
    evidence = claim_obj.get("evidence", {})
    if not evidence:
        return "NOT_ENOUGH_INFO"
    first_doc_evidence = next(iter(evidence.values()))
    raw_label = first_doc_evidence[0].get("label", "NOT_ENOUGH_INFO")
    return raw_label


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(CORPUS_PATH) as f:
        corpus_list = [json.loads(line) for line in f if line.strip()]
    with open(CLAIMS_PATH) as f:
        dev_claims = [json.loads(line) for line in f if line.strip()]

    corpus = {int(doc["doc_id"]): doc for doc in corpus_list}

    print(f"Loaded {len(corpus)} corpus docs, {len(dev_claims)} dev claims")

    print("Building BM25 index...")
    bm25 = BM25Retriever()
    bm25.build_index(corpus)

    print(f"Loading verifier: {CHECKPOINT_PATH}")
    nli_verifier = verifier.NLIVerifier()
    nli_verifier.load(str(CHECKPOINT_PATH))

    true_labels, preds= [], []
    numeric_corrections = 0

    for i, claim_obj in enumerate(dev_claims):
        claim_text = claim_obj["claim"]
        gold_label = resolve_gold_label(claim_obj)

        retrieved = bm25.retrieve(claim_text, top_k=K)  
        retrieved_docs = [corpus[doc_id] for doc_id, _ in retrieved]

        doc_results = [verify_claim_against_doc(nli_verifier, claim_text, doc) for doc in retrieved_docs]

        final = aggregate_across_docs(doc_results)
        label = final["label"]

        true_labels.append(LABEL2ID[gold_label])
        preds.append(LABEL2ID[label])

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(dev_claims)} claims...")

    true_labels = np.array(true_labels)
    preds = np.array(preds)

    def report(preds, name):
        acc = accuracy_score(true_labels, preds)
        macro_f1 = f1_score(true_labels, preds, average="macro")
        per_class = f1_score(true_labels, preds, average=None, labels=[0, 1, 2])
        print(f"\n{name} (end-to-end, k={K})")
        print(f"  accuracy:      {acc:.4f}")
        print(f"  macro_f1:      {macro_f1:.4f}")
        print(f"  contradict_f1: {per_class[0]:.4f}")
        print(f"  support_f1:    {per_class[1]:.4f}")
        print(f"  nei_f1:        {per_class[2]:.4f}")
        cm = confusion_matrix(true_labels, preds, labels=[0, 1, 2])
        print(f"  confusion matrix (rows=true, cols=pred, order=CONTRADICT/SUPPORT/NEI):")
        print(f"  {cm}")

    report(preds, "retrieval + verifier")