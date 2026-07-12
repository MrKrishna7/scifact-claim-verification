import re
from dataclasses import dataclass
from typing import Optional
from dataclasses import dataclass


_RAW_PAIRS: list[tuple[str, str]] = [
    ("increase",      "decrease"),
    ("increases",     "decreases"),
    ("increased",     "decreased"),
    ("increasing",    "decreasing"),
    ("elevate",       "lower"),
    ("elevates",      "lowers"),
    ("elevated",      "lowered"),
    ("elevating",     "lowering"),
    ("upregulate",    "downregulate"),
    ("upregulates",   "downregulates"),
    ("upregulated",   "downregulated"),
    ("upregulating",  "downregulating"),
    ("promote",       "inhibit"),
    ("promotes",      "inhibits"),
    ("promoted",      "inhibited"),
    ("promoting",     "inhibiting"),
    ("enhance",       "impair"),
    ("enhances",      "impairs"),
    ("enhanced",      "impaired"),
    ("enhancing",     "impairing"),
    ("improve",       "worsen"),
    ("improves",      "worsens"),
    ("improved",      "worsened"),
    ("improving",     "worsening"),
    ("activate",      "suppress"),
    ("activates",     "suppresses"),
    ("activated",     "suppressed"),
    ("activating",    "suppressing"),
    ("stimulate",     "block"),
    ("stimulates",    "blocks"),
    ("stimulated",    "blocked"),
    ("stimulating",   "blocking"),
    ("induce",        "prevent"),
    ("induces",       "prevents"),
    ("induced",       "prevented"),
    ("inducing",      "preventing"),
    ("accelerate",    "decelerate"),
    ("accelerates",   "decelerates"),
    ("accelerated",   "decelerated"),
    ("accelerating",  "decelerating"),
    ("amplify",       "attenuate"),
    ("amplifies",     "attenuates"),
    ("amplified",     "attenuated"),
    ("amplifying",    "attenuating"),
    ("facilitate",    "impede"),
    ("facilitates",   "impedes"),
    ("facilitated",   "impeded"),
    ("facilitating",  "impeding"),
    ("higher",        "lower"),
    ("greater",       "smaller"),
    ("larger",        "smaller"),
    ("faster",        "slower"),
    ("stronger",      "weaker"),
    ("longer",        "shorter"),
    ("earlier",       "later"),
    ("more",          "less"),
    ("positive",      "negative"),
    ("beneficial",    "harmful"),
    ("protective",    "detrimental"),
    ("superior",      "inferior"),
    ("effective",     "ineffective"),
    ("significant",   "nonsignificant"),
    ("associated",    "unassociated"),
    ("present",       "absent"),
    ("expressed",     "unexpressed"),
    ("active",        "inactive"),
    ("stable",        "unstable"),
    ("sensitive",     "resistant"),
    ("responsive",    "unresponsive"),
    ("sufficient",    "insufficient"),
]

ANTONYM_MAP: dict[str, str] = {}
for a, b in _RAW_PAIRS:
    ANTONYM_MAP[a] = b
    ANTONYM_MAP[b] = a

_CONTRACTIONS: list[tuple[str, str]] = [
    (r"\bcannot\b",    "can"),
    (r"\bwon't\b",     "will"),
    (r"\bcan't\b",     "can"),
    (r"\bdon't\b",     "do"),
    (r"\bdoesn't\b",   "does"),
    (r"\bdidn't\b",    "did"),
    (r"\bisn't\b",     "is"),
    (r"\baren't\b",    "are"),
    (r"\bwasn't\b",    "was"),
    (r"\bweren't\b",   "were"),
    (r"\bhasn't\b",    "has"),
    (r"\bhaven't\b",   "have"),
    (r"\bhadn't\b",    "had"),
    (r"\bshouldn't\b", "should"),
    (r"\bwouldn't\b",  "would"),
    (r"\bcouldn't\b",  "could"),
    (r"\bmightn't\b",  "might"),
]

_AUX_NOT = re.compile(
    r"\b(is|are|was|were|has|have|had|does|do|did"
    r"|can|could|will|would|should|may|might)\s+not\s+",
    re.IGNORECASE,
)

_NO_BEFORE_WORD = re.compile(r"\bno\s+(?=[a-zA-Z])", re.IGNORECASE)


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original and original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def remove_negation(text: str) -> Optional[str]:
    for pat, pos_aux in _CONTRACTIONS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return text[:m.start()] + _preserve_case(m.group(), pos_aux) + text[m.end():]

    m = _AUX_NOT.search(text)
    if m:
        return text[:m.start()] + m.group(1) + " " + text[m.end():]

    m = _NO_BEFORE_WORD.search(text)
    if m:
        result = (text[:m.start()] + text[m.end():]).strip() 
        if m.start() == 0 and result:
            result = result[0].upper() + result[1:]
        return result

    return None


def antonym_swap(text: str) -> Optional[str]:
    tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
    for i, tok in enumerate(tokens):
        antonym = ANTONYM_MAP.get(tok.lower())
        if antonym is not None:
            tokens[i] = _preserve_case(tok, antonym)
            return "".join(tokens)
    return None


def insert_negation(text: str) -> Optional[str]:

    _AUX_ORDER = [
        "is", "are", "was", "were", "has", "have", "had",
        "does", "do", "did", "can", "could", "will", "would",
        "should", "may", "might",
    ]
    for aux in _AUX_ORDER:
        m = re.search(rf"\b{aux}\b", text, re.IGNORECASE)
        if m:
            return text[:m.end()] + " not" + text[m.end():]
    return None
@dataclass
class NegationResult:
    original:   str
    negated:    Optional[str]
    rule:       Optional[str]

    @property
    def success(self) -> bool:
        return self.negated is not None

    def __repr__(self) -> str:
        if self.success:
            return f"[{self.rule}]\n  IN : {self.original}\n  OUT: {self.negated}"
        return f"[FAILED]\n  IN : {self.original}"


class ClaimNegator:

    def __init__(self, use_insert_negation: bool = True):
        self.use_insert_negation = use_insert_negation
        # because insert_negation is a more aggressive strategy and can sometimes produce less natural sentences.

    def negate(self, claim: str) -> Optional[str]:
        return self.negate_full(claim).negated

    def negate_full(self, claim: str) -> NegationResult:
        claim = claim.strip()

        for strategy_fn, name in [
            (remove_negation, "remove_negation"),
            (antonym_swap,    "antonym_swap"),
        ]:
            result = strategy_fn(claim)
            if result and result.lower() != claim.lower():
                return NegationResult(claim, result, name)

        if self.use_insert_negation:
            result = insert_negation(claim)
            if result and result.lower() != claim.lower():
                return NegationResult(claim, result, "insert_negation")

        return NegationResult(claim, None, None)