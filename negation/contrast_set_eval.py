import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import  Optional

from models import verifier

from .claim_negator import ClaimNegator

CLAIMS_PATH = Path("data/claims_dev.jsonl")
CORPUS_PATH = Path("data/corpus.jsonl")
CHECKPOINT_PATH = Path("models/saved/verifier_unweighted")
OUT_PATH = Path("contrast_results.json")

USE_INSERT_NEGATION = True
VERBOSE = True


EXPECTED_FLIP = {"SUPPORT": "CONTRADICT", "CONTRADICT": "SUPPORT"}

def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_corpus_index(path: str | Path) -> dict[int, list[str]]:
    corpus = load_jsonl(path)
    index = {}

    for doc in corpus:
        doc_id = doc.get("doc_id")
        if doc_id is None:
            doc_id = doc.get("paper_id")
        if doc_id is None:
            continue

        index[int(doc_id)] = doc.get("abstract", [])

    return index

def extract_gold_pairs(claims: list[dict], corpus_index: dict[int, list[str]]) -> list[dict]:
    pairs = []
    for claim in claims:
        for doc_id, evidence_list in (claim.get("evidence") or {}).items():
            doc_id = int(doc_id)
            abstract = corpus_index.get(doc_id)
            for ev in evidence_list:
                label = ev.get("label")
                if label not in ("SUPPORT", "CONTRADICT"):
                    continue
                evidence_text = " ".join(
                    abstract[i] for i in ev.get("sentences", []) if 0 <= i < len(abstract)
                )
                if not evidence_text:
                    continue
                pairs.append({
                    "claim_id": int(claim["id"]), "claim": claim["claim"],
                    "label": label, "doc_id": doc_id, "evidence": evidence_text,
                })
    return pairs


@dataclass
class ContrastResult:
    claim_id: int
    original_claim: str
    negated_claim: Optional[str]
    negation_rule: Optional[str]
    gold_label: str
    pred_original: Optional[str]
    pred_negated: Optional[str]
    conf_original: Optional[float]
    conf_negated: Optional[float]
    outcome: str  # flip | consistency_failure | wrong_direction | no_negation | error
    pred_original_correct: Optional[bool] = None

def predict(verifier, claim: str, evidence: str):
    doc = {"abstract": [evidence]}
    output = verifier.predict(claim, doc)
    return output["label"], output["confidence"]

def evaluate_pair(verifier, pair: dict, negator: ClaimNegator) -> ContrastResult:
    claim, evidence, gold_label = pair["claim"], pair["evidence"], pair["label"]
    neg = negator.negate_full(claim)

    base = dict(claim_id=pair["claim_id"], original_claim=claim, gold_label=gold_label)

    if not neg.success or neg.negated is None:
        return ContrastResult(
            **base,
            negated_claim=None,
            negation_rule=None,
            pred_original=None,
            pred_negated=None,
            conf_original=None,
            conf_negated=None,
            outcome="no_negation",
            pred_original_correct=None,
        )

    pred_orig, conf_orig = predict(verifier, claim, evidence)
    pred_neg, conf_neg = predict(verifier, neg.negated, evidence)

    expected = EXPECTED_FLIP[gold_label]

    if pred_orig == pred_neg:
        outcome = "consistency_failure"
    elif pred_neg == expected:
        outcome = "flip"
    else:
        outcome = "wrong_direction"   

    return ContrastResult(
        **base,
        negated_claim=neg.negated,
        negation_rule=neg.rule,
        pred_original=pred_orig,
        pred_negated=pred_neg,
        conf_original=conf_orig,
        conf_negated=conf_neg,
        outcome=outcome,
        pred_original_correct=(pred_orig == gold_label),
    )

def pct(a: int, b: int) -> float:
    return round(100 * a / b, 2) if b else 0.0

def load_verifier():
    v = verifier.NLIVerifier()
    v.load(str(CHECKPOINT_PATH))
    return v

def run_contrast_eval( claims_path, corpus_path, use_insert_negation=True, verbose=True):
    negator = ClaimNegator(use_insert_negation=use_insert_negation)
    corpus = build_corpus_index(corpus_path)
    pairs = extract_gold_pairs(load_jsonl(claims_path), corpus)

    print(f"Loaded {len(pairs)} gold pairs.")

    results = []
    for i, pair in enumerate(pairs):
        if verbose and (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(pairs)}", end="\r")
        results.append(evaluate_pair(verifier, pair, negator))

    evaluated = [r for r in results if r.outcome not in ("error", "no_negation")]
    flips = [r for r in evaluated if r.outcome == "flip"]
    failures = [r for r in evaluated if r.outcome == "consistency_failure"]
    wrong = [r for r in evaluated if r.outcome == "wrong_direction"]
    correct_original = [r for r in evaluated if r.pred_original_correct]
    flip_given_correct = [r for r in correct_original if r.outcome == "flip"]

    summary = {
        "total_pairs": len(results),
        "evaluated_pairs": len(evaluated),
        "negation_coverage": pct(len(evaluated), len(results)),
        "flip_rate": pct(len(flips), len(evaluated)),
        "consistency_failure": pct(len(failures), len(evaluated)),
        "wrong_direction": pct(len(wrong), len(evaluated)),
        "flip_rate_given_correct_original": pct(len(flip_given_correct), len(correct_original)),
    }

    print("\nCONTRAST SET RESULTS\n")
    for k, v in summary.items():
        print(f"{k:35s}: {v}")

    return {"summary": summary, "results": [asdict(r) for r in results],
            "failures": [asdict(r) for r in failures]}


if __name__ == "__main__":
    verifier = load_verifier();
    report = run_contrast_eval(CLAIMS_PATH, CORPUS_PATH)

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {out_path.resolve()}")
