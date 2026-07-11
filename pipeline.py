from typing import List, Dict

from models.retriever import  BM25Retriever
from models.verifier import NLIVerifier
from utils.data_utils import load_corpus


class ScifactPipeline:
    def __init__(
        self,
        corpus_path: str = "data/corpus.jsonl",
        top_k: int = 3,
        nei_threshold: float = 0.5,
    ):
        self.corpus_path = corpus_path
        self.top_k = top_k
        self.nei_threshold = nei_threshold
        self.corpus = None
        self.retriever = None
        self.verifier = None

    def load(
        self,
        bm25_index_path: str = "data/bm25_index.pkl",
        verifier_path: str = "models/saved/verifier_unweighted",  
    ):
        print("Loading corpus...")
        self.corpus = load_corpus(self.corpus_path)
        bm25 = BM25Retriever()
        bm25.load_index(bm25_index_path)
        self.retriever = bm25;
        print("Loading verifier...")
        self.verifier = NLIVerifier()

        self.verifier.load(verifier_path)

        print("Pipeline ready.")

    def verify(self, claim: str) -> Dict:

        retrieved = self.retriever.retrieve(claim, top_k=self.top_k)

        evidence_list = []
        for doc_id, retrieval_score in retrieved:
            doc = self.corpus[doc_id]

            result = self.verifier.predict(claim, doc, threshold_nei=self.nei_threshold)

            evidence_list.append({
                "doc_id": doc_id,
                "title": doc["title"],
                "label": result["label"],
                "confidence": result["confidence"],
                "evidence_sentence": result["evidence_sentence"],
                "evidence_index": result["evidence_index"],
                "evidence_sentences": result.get("evidence_sentences", []),
                "retrieval_score": retrieval_score,
                "sentence_scores": result["sentence_scores"],
            })

        best = None
        for ev in evidence_list:
            if ev["label"] != "NOT_ENOUGH_INFO":
                if best is None or ev["confidence"] > best["confidence"]:
                    best = ev

        if best is None:
            verdict = "NOT_ENOUGH_INFO"
            nei_confs = [
                ev["sentence_scores"][0].get("NOT_ENOUGH_INFO", 0.0)
                for ev in evidence_list
                if ev.get("sentence_scores")
            ]
            confidence = float(sum(nei_confs) / len(nei_confs)) if nei_confs else 0.0
        else:
            verdict = best["label"]
            confidence = float(best["confidence"])

        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "evidence": evidence_list,
        }

    def verify_batch(self, claims: List[str]) -> List[Dict]:
        return [self.verify(claim) for claim in claims]


if __name__ == "__main__":
    pipe = ScifactPipeline(top_k=3) 
    pipe.load()  

    test_claims = [
        "1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.",
        "1/2000 in UK have abnormal PrP positivity.", 
        # 493/1,000,000=0.000493
        # which is about 1 in 2,028
        "Less than 10% of the gabonese children with Schimmelpenning-Feuerstein-Mims syndrome (SFM) had a plasma lactate of more than 5mmol/L.",
        "ALDH1 expression is associated with better breast cancer outcomes.",
        "Smoking causes lung cancer."
    ]

    for claim in test_claims:
        result = pipe.verify(claim)
        print(f"\nClaim     : {result['claim']}")
        print(f"Verdict   : {result['verdict']} (conf={result['confidence']:.3f})")
        if result["evidence"]:
            best = result["evidence"][0]
            print(f"Evidence  : {best['evidence_sentence']}")