import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_utils import load_corpus, load_claims
from models.retriever import DenseRetriever, BM25Retriever, HybridRetriever


def evaluate_recall(retriever, claims, top_k_list=(3, 5, 10), split_name="dev"):
    eval_claims = [c for c in claims if c.get("evidence", {})]
    total_claims_with_evidence = len(eval_claims)
    hits = {k: 0 for k in top_k_list}

    if total_claims_with_evidence == 0:
        print(f"\n[{split_name}] No claims with evidence found.")
        return {f"recall@{k}": 0 for k in top_k_list}

    max_k = max(top_k_list)
    claim_texts = [c["claim"] for c in eval_claims]
    all_results = retriever.retrieve_batch(claim_texts, top_k=max_k)
    for claim, results in zip(eval_claims, all_results):
        gold_doc_ids = {int(doc_id) for doc_id in claim["evidence"].keys()}
        for k in top_k_list:
            retrieved_k = {doc_id for doc_id, _ in results[:k]}
            if gold_doc_ids & retrieved_k:
                hits[k] += 1

    print(f"\n[{split_name}] Recall@K  (over {total_claims_with_evidence} claims with evidence)")
    for k in top_k_list:
        r = hits[k] / total_claims_with_evidence if total_claims_with_evidence else 0
        print(f"  Recall@{k:<3} = {r:.4f}  ({hits[k]}/{total_claims_with_evidence})")

    return {
        f"recall@{k}": hits[k] / total_claims_with_evidence if total_claims_with_evidence else 0
        for k in top_k_list
    }


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    corpus = load_corpus(os.path.join(data_dir, "corpus.jsonl"))
    dev_claims = load_claims(os.path.join(data_dir, "claims_dev.jsonl"))

    print(f"Corpus size: {len(corpus)} documents")

    dense = DenseRetriever()
    if not os.path.exists(DenseRetriever.INDEX_PATH):
        dense.build_index(corpus, save_path=DenseRetriever.INDEX_PATH)
    else:
        print("Dense index already exists. Loading...")
        dense.load_index(DenseRetriever.INDEX_PATH)

    bm25 = BM25Retriever()
    if not os.path.exists(BM25Retriever.INDEX_PATH):
        bm25.build_index(corpus, save_path=BM25Retriever.INDEX_PATH)
    else:
        print("BM25 index already exists. Loading...")
        bm25.load_index(BM25Retriever.INDEX_PATH)

    print("RETRIEVER COMPARISON ON DEV SET")

    print("Dense (SPECTER + FAISS)")
    dense_scores = evaluate_recall(dense, dev_claims, split_name="dev")

    print("BM25")
    bm25_scores = evaluate_recall(bm25, dev_claims, split_name="dev")

    hybrid = HybridRetriever(dense, bm25)
    print("Hybrid")
    hybrid_scores = evaluate_recall(hybrid, dev_claims, split_name="dev")

    print("SUMMARY")
    print(f"{'':20} {'@3':>8} {'@5':>8} {'@10':>8}")

    for name, scores in [("Dense", dense_scores), ("BM25", bm25_scores), ("Hybrid", hybrid_scores)]:
        print(f"{name:<20} {scores['recall@3']:>8.4f} {scores['recall@5']:>8.4f} {scores['recall@10']:>8.4f}")

    results = {
        "dense": dense_scores,
        "bm25": bm25_scores,
        "hybrid": hybrid_scores,
    }
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/retrieval_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to evaluation/retrieval_comparison.json")


