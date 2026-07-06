import json
import os
import random
from typing import Any, Dict, List, Tuple
from collections import Counter

LABEL2ID = {
    "SUPPORT": 0,
    "CONTRADICT": 1,
    "NOT_ENOUGH_INFO": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _choose_sentence(
    doc: Dict,
    exclude_indices: set[int] | None = None,
    rng: random.Random | None = None,
) -> str | None:
    rng = rng or random
    exclude_indices = exclude_indices or set()

    choices = []
    for i, sent in enumerate(doc.get("abstract", [])):
        if i in exclude_indices:
            continue
        if sent:
            choices.append(sent)

    if not choices:
        return None
    return rng.choice(choices)


def load_corpus(corpus_path: str) -> Dict[int, Dict]:
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            corpus[doc["doc_id"]] = {
                "title": doc.get("title", ""),
                "abstract": doc.get("abstract", []),
            }
    return corpus


def get_abstract_text(doc: Dict, include_title: bool = True) -> str:
    parts = []
    if include_title and doc.get("title"):
        parts.append(doc["title"])
    parts.extend(doc.get("abstract", []))
    return " ".join(parts).strip()


def load_claims(claims_path: str) -> List[Dict]:
    claims = []
    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                claims.append(json.loads(line))
    return claims


def get_claim_label(claim: Dict, doc_id: int) -> str:
    evidence = claim.get("evidence", {}) or {}
    ev_value = evidence.get(str(doc_id))
    ev_info = ev_value[0]

    if ev_info is None:
        return "NOT_ENOUGH_INFO"

    label = str(ev_info.get("label", "")).strip().upper()
    if label in LABEL2ID:
        return label

    return "NOT_ENOUGH_INFO"


def get_evidence_sentences(claim: Dict, doc_id: int) -> List[int]:
    evidence = claim.get("evidence", {}) or {}
    ev_value = evidence.get(str(doc_id))
    ev_info =ev_value[0]

    if ev_info is None:
        return []

    sents = ev_info.get("sentences", [])
    return sents if isinstance(sents, list) else []


def build_nli_pairs(
    claims: List[Dict],
    corpus: Dict[int, Dict],
    use_gold_sentences: bool = True,
    nei_pairs_per_claim: int = 2,
    hard_negatives_per_positive: int = 1,
) -> List[Dict]:
    rng = random.Random(42)
    pairs = []
    all_doc_ids = list(corpus.keys())
    def add_nei_pair(claim_text: str, claim_id: int, doc_id: int, source: str):
        if doc_id not in corpus:
            return
        premise = _choose_sentence(corpus[doc_id], rng=rng)
        if not premise:
            return
        pairs.append({
            "premise": premise,
            "hypothesis": claim_text,
            "label": LABEL2ID["NOT_ENOUGH_INFO"],
            "claim_id": claim_id,
            "doc_id": doc_id,
            "source": source,
        })
    def add_random_nei(claim_text: str, claim_id: int, excluded: set[int], k: int):
        if k <= 0 or not all_doc_ids:
            return

        candidates = [d for d in all_doc_ids if d not in excluded]
        if not candidates:
            candidates = all_doc_ids[:]

        sample_k = min(k, len(candidates))
        for doc_id in rng.sample(candidates, sample_k):
            add_nei_pair(claim_text, claim_id, doc_id, "random_nei")



    

    for claim in claims:
        claim_text = claim["claim"]
        claim_id = claim["id"]
        evidence = claim.get("evidence", {}) or {}
        cited_doc_ids = claim.get("cited_doc_ids", []) or []
        evidenced_doc_ids = {int(doc_id) for doc_id in evidence.keys()}

        # Claims with no evidence anywhere are NEI.
        if not evidence:
            used = 0
            for doc_id in cited_doc_ids:
                if used >= nei_pairs_per_claim:
                    break
                if doc_id in corpus:
                    add_nei_pair(claim_text, claim_id, doc_id, "cited_doc_nei")
                    used += 1

            if used < nei_pairs_per_claim:
                excluded = set(cited_doc_ids)
                add_random_nei(claim_text, claim_id, excluded, nei_pairs_per_claim - used)

            continue

        # Claims with labeled evidence.
        for doc_id_str, ev_value in evidence.items():
            doc_id = int(doc_id_str)
            if doc_id not in corpus:
                continue

            ev_info = ev_value[0]
            if ev_info is None:
                continue

            label = str(ev_info.get("label", "")).strip().upper()
            if label not in ("SUPPORT", "CONTRADICT"):
                continue

            doc = corpus[doc_id]
            sent_indices = ev_info.get("sentences", [])

            if use_gold_sentences and sent_indices:
                evidence_sents = [
                    doc["abstract"][i]
                    for i in sent_indices
                    if 0 <= i < len(doc["abstract"])
                ]
                premise = " ".join(evidence_sents).strip()
            else:
                premise = get_abstract_text(doc)

            if premise:
                pairs.append({
                    "premise": premise,
                    "hypothesis": claim_text,
                    "label": LABEL2ID[label],
                    "claim_id": claim_id,
                    "doc_id": doc_id,
                })

            if use_gold_sentences and sent_indices:
                exclude = set(sent_indices)
                for _ in range(max(0, hard_negatives_per_positive)):
                    hard_premise = _choose_sentence(doc, exclude_indices=exclude, rng=rng)
                    if hard_premise:
                        pairs.append({
                            "premise": hard_premise,
                            "hypothesis": claim_text,
                            "label": LABEL2ID["NOT_ENOUGH_INFO"],
                            "claim_id": claim_id,
                            "doc_id": doc_id,
                            "source": "hard_negative_same_doc",
                        })

        cited_but_unevidenced = [d for d in cited_doc_ids if d not in evidenced_doc_ids]
        for doc_id in cited_but_unevidenced:
            add_nei_pair(claim_text, claim_id, doc_id, "cited_doc_nei")

        excluded = set(cited_doc_ids) | evidenced_doc_ids
        add_random_nei(claim_text, claim_id, excluded, 1)

    return pairs


def print_label_distribution(pairs: List[Dict]):
    

    if not pairs:
        print("\nLabel distribution: no pairs created.\n")
        return

    counts = Counter(ID2LABEL[p["label"]] for p in pairs)
    total = sum(counts.values())

    print("\nLabel distribution:")
    for label, count in sorted(counts.items()):
        print(f"  {label:<18} {count:>5}  ({100 * count / total:.1f}%)")
    print(f"  {'TOTAL':<18} {total:>5}\n")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    train_path = os.path.join(data_dir, "claims_train.jsonl")

    if not os.path.exists(corpus_path):
        print("Run `python utils/download_data.py` first.")
        raise SystemExit(1)

    corpus = load_corpus(corpus_path)
    claims = load_claims(train_path)
    pairs = build_nli_pairs(claims, corpus)

    print(f"Corpus docs  : {len(corpus)}")
    print(f"Train claims : {len(claims)}")
    print(f"NLI pairs    : {len(pairs)}")
    print_label_distribution(pairs)

    if pairs:
        sample = pairs[43]
        print("Sample NLI pair:")
        print("Premise   :", sample["premise"][:160])
        print("Hypothesis:", sample["hypothesis"][:160])
        print("Label     :", ID2LABEL[sample["label"]])