import os
from typing import List, Dict, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class NLIVerifier:
    PRETRAINED_MODEL = "cross-encoder/nli-deberta-v3-base"
    

    def __init__(self, model_name: str = PRETRAINED_MODEL, device: str = "auto"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)

    def _load_model(self):
        if self.model is not None:
            return
        print(f"Loading verifier: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict_pair(
        self,
        premise: str,
        hypothesis: str,
        max_length: int = 512,
    ) -> Dict[str, float]:

        self._load_model()
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            #   - pretrained NLI labels: entailment / contradiction / neutral
            #   - fine-tuned labels: SUPPORT / CONTRADICT / NOT_ENOUGH_INFO
            
            # contradiction=0, entailment=1, neutral=2
            labels = ["CONTRADICT", "SUPPORT", "NOT_ENOUGH_INFO"]

        result: Dict[str, float] = {}
        for idx, prob in enumerate(probs):
            our_label = labels[idx]
            result[our_label] = float(prob)


        return result

    def predict_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 16,
    ) -> List[Dict[str, float]]:

        # Batch prediction for efficiency.

        self._load_model()
        all_results: List[Dict[str, float]] = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            premises = [p[0] for p in batch]
            hypotheses = [p[1] for p in batch]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()

            # contradiction=0, entailment=1, neutral=2
            labels = ["CONTRADICT", "SUPPORT", "NOT_ENOUGH_INFO"]
            for probs in probs_batch:
                result: Dict[str, float] = {}
                for idx, prob in enumerate(probs):
                    our_label = labels[idx]
                    result[our_label] =  float(prob)

                all_results.append(result)

        return all_results

    def predict(
        self,
        claim: str,
        doc: Dict,
        threshold_nei: float = 0.5,
        evidence_threshold: float = 0.35,
        max_evidence_sentences: int = 3,
    ) -> Dict:
        
        sentences = doc.get("abstract", [])
        if not sentences:
            return {
                "label": "NOT_ENOUGH_INFO",
                "confidence": 0.0,
                "evidence_sentence": None,
                "evidence_index": None,
                "evidence_sentences": [],
                "sentence_scores": [],
            }

        pairs = [(sent, claim) for sent in sentences]
        scores = self.predict_batch(pairs)

        sentence_scores = []
        for sent, score_dict in zip(sentences, scores):
            score = {
                "Text": sent,
                "SUPPORT": score_dict.get("SUPPORT", 0.0),
                "CONTRADICT": score_dict.get("CONTRADICT", 0.0),
                "NOT_ENOUGH_INFO": score_dict.get("NOT_ENOUGH_INFO", 0.0),
            }
            score["Relevance"] = max(score["SUPPORT"], score["CONTRADICT"])
            sentence_scores.append(score)

        # Best evidence sentence
        best_idx = max(range(len(sentence_scores)), key=lambda i: sentence_scores[i]["Relevance"])
        best = sentence_scores[best_idx]

        # Keeping the top-N evidence candidates above threshold, sorted by Relevance
        ranked = sorted(
            enumerate(sentence_scores),
            key=lambda item: item[1]["Relevance"],
            reverse=True,
        )
        evidence_indices = [
            idx for idx, score in ranked
            if score["Relevance"] >= evidence_threshold
        ][:max_evidence_sentences]

        support_score = best["SUPPORT"]
        contradict_score = best["CONTRADICT"]
        nei_score = best["NOT_ENOUGH_INFO"]
        max_sr = max(support_score, contradict_score)

        if max_sr < threshold_nei:
            label = "NOT_ENOUGH_INFO"
            confidence = nei_score
        elif support_score >= contradict_score:
            label = "SUPPORT"
            confidence = support_score
        else:
            label = "CONTRADICT"
            confidence = contradict_score

        return {
            "label": label,
            "confidence": float(confidence),
            "evidence_sentence": sentences[best_idx],
            "evidence_index": best_idx,
            "evidence_sentences": evidence_indices,
            "sentence_scores": sentence_scores,
        }

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Verifier saved to {save_dir}")

    def load(self, load_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_dir)
        self.model.to(self.device)
        self.model.eval()
        print(f"Verifier loaded from {load_dir}")
