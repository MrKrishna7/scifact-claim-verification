import random
from typing import Optional

from .claim_negator import ClaimNegator

FLIP = { "CONTRADICT": "SUPPORT","SUPPORT": "CONTRADICT"}


def augment_nli_pairs(
    pairs:                list[dict],
    ratio:                float = 1.0,
    use_insert_negation:  bool  = False,
    shuffle:              bool  = True,
    seed:                 int   = 42,
    label_map:            Optional[dict] = None,
) -> list[dict]:
    negator = ClaimNegator(use_insert_negation=use_insert_negation)
    rng     = random.Random(seed)

    original_count  = len(pairs)
    generated       = []
    skipped_nei     = 0
    skipped_no_neg  = 0

    for pair in pairs:
        label     = pair.get("label")
        hypothesis = pair.get("hypothesis") or pair.get("claim", "")

        inv_map = {v: k for k, v in label_map.items()}
        str_label = inv_map.get(label, label)
        if str_label not in FLIP:
            skipped_nei += 1
            continue

        neg_result = negator.negate_full(hypothesis)
        if not neg_result.success:
            skipped_no_neg += 1
            continue

        flipped_label_str = FLIP[str_label]
        flipped_label = label_map[flipped_label_str]

        new_pair = {
            **pair,
            "hypothesis":    neg_result.negated,
            "label":         flipped_label,
            "_augmented":    True,      # flag
            "_negation_rule": neg_result.rule,
            "_original_hypothesis": hypothesis,
        }
        generated.append(new_pair)
    if ratio <= 0.0:
        generated = []
    elif ratio < 1.0 and generated:
        k = min(len(generated), max(1, int(len(generated) * ratio)))
        generated = rng.sample(generated, k)

    augmented_pairs = list(pairs) + generated

    if shuffle:
        rng.shuffle(augmented_pairs)

    print(
        f"[augment_nli_pairs] "
        f"original={original_count}  "
        f"generated={len(generated)}  "
        f"skipped_nei={skipped_nei}  "
        f"skipped_no_neg={skipped_no_neg}  "
        f"total={len(augmented_pairs)}"
    )
        # Rule breakdown
    rule_counts: dict[str, int] = {}
    for p in generated:
        r = p.get("_negation_rule", "unknown")
        rule_counts[r] = rule_counts.get(r, 0) + 1
    print(f"  Negation rule breakdown: {rule_counts}")

    return augmented_pairs