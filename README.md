# SciFact Claim Verification

A two-stage scientific claim verification pipeline (retrieval → NLI verification) built on the SciFact dataset, with a dedicated diagnostic suite for measuring and improving polarity/negation robustness — a known weak point in NLI-based fact verifiers.

## Overview

Given a scientific claim, the pipeline retrieves candidate evidence abstracts from a corpus and classifies the claim as **SUPPORT**, **CONTRADICT**, or **NOT_ENOUGH_INFO (NEI)** relative to that evidence. Beyond the base pipeline, this project specifically investigates and addresses a failure mode common to NLI verifiers: insensitivity to claim polarity (i.e., failing to flip predictions correctly when a claim is logically negated).

## Architecture

| Component    | Choice                         | Why                                                                                                                                                  |
| ------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Corpus       | ~5,000 scientific abstracts    | Small, closed corpus — retrieval is a comparatively easier sub-problem here than in open-domain settings                                             |
| Retrieval    | **BM25**                       | Benchmarked against Dense (SPECTER) and Hybrid (RRF fusion); BM25 alone outperformed both at every k — see [Retrieval Ablation](#retrieval-ablation) |
| Verifier     | Fine-tuned **DeBERTa-v3-base** | NLI classification head over SUPPORT / CONTRADICT / NEI                                                                                              |
| Augmentation | **ClaimNegator**               | Rule-based negation generator (antonym substitution, negation removal, negation insertion) used to build contrast pairs for training and evaluation  |

### Why BM25 over Dense/Hybrid

Dense retrieval (SPECTER) underperformed BM25 substantially at every k, and Hybrid (BM25+Dense via reciprocal rank fusion) underperformed BM25 alone as well — likely because the weak dense signal introduces noise into the fused ranking rather than complementary signal, on a domain (scientific claims) where precise terminology overlap matters more than semantic smoothing. SPECTER2 was also tested as a drop-in replacement for SPECTER and found ~5% worse on recall@3, so the original SPECTER embeddings were kept.

## Dataset

| Label           | Count     | %     |
| --------------- | --------- | ----- |
| NOT_ENOUGH_INFO | 1,767     | 66.6% |
| SUPPORTS        | 511       | 19.2% |
| REFUTES         | 377       | 14.2% |
| **Total**       | **2,655** |       |

Class imbalance (NEI dominating ~2:1 over SUPPORTS+REFUTES combined) is a persistent factor across every evaluation below.

## Key Results

### Retrieval Ablation (recall@k)

|                 | @3         | @5         | @10        |
| --------------- | ---------- | ---------- | ---------- |
| Dense (SPECTER) | 0.3777     | 0.4681     | 0.5851     |
| **BM25**        | **0.7872** | **0.8351** | **0.8989** |
| Hybrid (RRF)    | 0.7021     | 0.8298     | 0.8883     |

BM25 wins outright at every k. Production retriever is BM25-only.

### Verifier Performance — Negation-Augmentation Ablation (oracle evidence)

Isolating the effect of ClaimNegator's negation rule coverage on verifier quality:

| Config                          | Train pairs | Accuracy | Macro F1 | Contradict F1 | Support F1 | NEI F1 |
| ------------------------------- | ----------- | -------- | -------- | ------------- | ---------- | ------ |
| No augmentation (baseline)      | 2,266       | 0.8890   | 0.7767   | 0.6418        | 0.7426     | 0.9457 |
| + remove_negation, antonym_swap | 2,555       | 0.9020   | 0.8143   | 0.7092        | 0.7865     | 0.9471 |
| + insert_negation (full)        | 2,671       | 0.9020   | 0.8224   | **0.7463**    | 0.7774     | 0.9436 |

Contradict F1 improves monotonically (+10.5pt total) as negation rule diversity increases, with accuracy and NEI F1 essentially stable — i.e., the gain is not coming at the expense of the majority class.

### Contrast-Set Consistency (Polarity Robustness)

The core diagnostic: for each SUPPORT/REFUTE claim, generate its negated counterpart via ClaimNegator and check whether the verifier's prediction flips correctly. Same 338 gold pairs, same evaluation script, before vs. after negation-augmented training:

| Metric                                     | Before augmentation | After augmentation | Δ       |
| ------------------------------------------ | ------------------- | ------------------ | ------- |
| Flip rate (correct polarity flip)          | 52.47%              | **61.60%**         | +9.13pt |
| Consistency failure (no change at all)     | 35.74%              | **30.04%**         | −5.70pt |
| Wrong direction (changed, but incorrectly) | 11.79%              | **8.37%**          | −3.42pt |
| Flip rate given original was correct       | 81.53%              | **85.80%**         | +4.27pt |

Negation-augmented training measurably improves polarity sensitivity across every metric. That said, ~30% of negated claims still fail to change the model's prediction at all — augmentation helps but does not fully resolve the underlying polarity-insensitivity.

### End-to-End Pipeline (Retrieval + Verifier, Real Evidence)

|                                               | Accuracy | Macro F1 | Contradict F1 | Support F1 | NEI F1 |
| --------------------------------------------- | -------- | -------- | ------------- | ---------- | ------ |
| Oracle evidence (verifier only)               | 0.9020   | 0.8224   | 0.7463        | 0.7774     | 0.9436 |
| **Full pipeline** (retrieval + verifier, k=5) | 0.6467   | 0.6318   | 0.5734        | 0.7259     | 0.5960 |

Confusion matrix (rows = true, cols = predicted; order CONTRADICT/SUPPORT/NEI):

```
[[41 10 13]
 [16 94 14]
 [22 31 59]]
```

The ~26-point accuracy gap between oracle and end-to-end evaluation is the dominant open question in this project — BM25 retrieval recall@5 (0.8351) accounts for only part of the drop, suggesting the remainder likely comes from sentence-level evidence selection within correctly-retrieved documents rather than missing documents outright. The confusion matrix also shows the pipeline's largest single error block is NEI claims being misclassified as SUPPORT (31 of 112 true-NEI cases) — a distinct failure mode from the polarity/negation weakness this project otherwise focuses on.

## Known Limitations

- **Oracle-vs-end-to-end gap not fully diagnosed.** Root cause (missing evidence vs. incorrect sentence selection within retrieved documents) is hypothesized but not yet isolated via manual error audit. BM25 recall@5 (0.8351) accounts for only part of the ~26-point accuracy drop between oracle and full-pipeline evaluation.

- **No external benchmark validation.** All results are on the closed, ~5K-abstract SciFact corpus; generalization to SciFact-Open (500K abstracts) is untested.

- **ClaimNegator negation coverage is ~78%**, not 100% — some claims can't be negated by the current rule set (antonym_swap / remove_negation / insert_negation) and are excluded from contrast-set evaluation.

- **Rule-based negation can produce scope errors.** A negation can be grammatically valid without asserting the true logical opposite. For example, negating "Children who fail to engage in regular physical activity are at greater risk of obesity" by inserting a negation near "fail to engage" can produce "Children who do not fail to engage..." — grammatically fine, but this does not actually assert the opposite of the original claim (the negation lands on the wrong scope). This is a known risk with naive rule-based negators: it can silently inject mislabeled training pairs (labeled CONTRADICT but not actually a logical contradiction), which could suppress contradict-class F1 gains rather than help them. The current rule set is not fully audited against this failure mode.

- **`insert_negation` picks the first auxiliary verb by fixed priority order, not the earliest one in the sentence.** In claims containing more than one auxiliary verb (e.g. "which was conducted... is associated with..."), this can insert the negation into the wrong clause rather than the main claim. Not yet fixed or measured for impact.

- **Numeric/approximate-value equivalence is a known, unaddressed failure mode.** The verifier can assign high confidence (e.g. 0.98) to CONTRADICT for claims where the numbers are actually equivalent across surface forms — e.g. treating "1/2000" and "493 per million" as unrelated and inferring contradiction, when they are approximately the same value. Manual inspection confirmed the retriever was working correctly in the case that surfaced this (the gold evidence sentence was retrieved) — the failure is specifically in the verifier's numeric reasoning, since DeBERTa relies on learned language patterns rather than performing explicit arithmetic, and treats differently-formatted numbers as unrelated tokens. **No mitigation has been implemented for this yet** — an earlier numeric-consistency layer was removed from the pipeline; the issue is documented here as a diagnosed but open problem, not a solved one.

- **Class imbalance limits reliability of minority-class metrics.** NEI outnumbers SUPPORT+REFUTES combined by roughly 2:1 in the training data (see [Dataset](#dataset)), and REFUTES is the smallest class. Contradict/refute F1 numbers throughout this project should be read with this imbalance in mind — smaller class sample sizes make these scores noisier than the NEI F1 numbers.

- **NEI-over-prediction as SUPPORT** is a known, unaddressed error mode in the full pipeline (see confusion matrix above) — 31 of 112 true-NEI claims are misclassified as SUPPORT, the largest single error block in the end-to-end confusion matrix.

## Possible Extensions

- **SciFact-Open generalization test** — evaluate the full pipeline against the 500K-abstract open corpus to check whether current results hold outside the easier closed-corpus setting. Expected to be the highest-value next step: it's a direct, citable comparison against a named external benchmark rather than another internal ablation.

## Repository Structure

```
scifact-claim-verification/
├── evaluation/     # contrast_set_eval.py, two-stage pipeline eval, retrieval ablation scripts
├── models/         # verifier (DeBERTa-v3-base NLI), retriever (BM25/Dense/Hybrid), claim_negator
├── utils/          # data loading and preprocessing helpers
└── requirements.txt
```
