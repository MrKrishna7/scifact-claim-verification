import os
import pickle
from typing import List, Tuple, Dict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from utils.data_utils import get_abstract_text
from rank_bm25 import BM25Okapi

class DenseRetriever:
    MODEL_NAME = "allenai/specter"
    INDEX_PATH = "data/faiss_index.pkl"

    def __init__(self, model_name: str = MODEL_NAME, device: str = "auto"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.doc_ids = None
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    def _load_model(self):
        if self.model is not None:
            return
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def _encode(self, texts, batch_size=64):
        self._load_model()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def build_index(self, corpus: Dict, save_path: str = INDEX_PATH):
    
        print(f"Building index for {len(corpus)} documents...")

        doc_ids = list(corpus.keys())
        texts = [get_abstract_text(corpus[doc_id], include_title=True) for doc_id in doc_ids]
        embeddings = self._encode(texts)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        print(f"Index built: {index.ntotal} vectors, dim={dim}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"index": index, "doc_ids": doc_ids}, f)

        self.index = index
        self.doc_ids = doc_ids
        print(f"Saved index to {save_path}")

    def load_index(self, load_path: str = INDEX_PATH):
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        self.index = data["index"]
        self.doc_ids = data["doc_ids"]
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")

    def retrieve(self, claim: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.index is None:
            raise AssertionError("Call build_index() or load_index() first")

        query_emb = self._encode([claim])
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append((self.doc_ids[idx], float(score)))
        return results

    def retrieve_batch(self, claims: List[str], top_k: int = 5) -> List[List[Tuple[int, float]]]:
        if self.index is None:
            raise AssertionError("Call build_index() or load_index() first")
        query_embs = self._encode(claims)
        scores_batch, indices_batch = self.index.search(query_embs, top_k)

        all_results = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = [
                (self.doc_ids[idx], float(score))
                for score, idx in zip(scores, indices)
            ]
            all_results.append(results)
        return all_results

class BM25Retriever:
    INDEX_PATH = "data/bm25_index.pkl"

    def __init__(self):
        self.bm25 = None
        self.doc_ids = None

    def _tokenize(self, text: str) -> List[str]:
         return text.lower().split()     
    # might use nltk or spacy later

    def build_index(self, corpus: Dict, save_path: str = INDEX_PATH):

        print("Building BM25 index...")
        self.doc_ids = list(corpus.keys())
        tokenized = [self._tokenize(get_abstract_text(corpus[doc_id], include_title=True)) for doc_id in self.doc_ids]
        self.bm25 = BM25Okapi(tokenized)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "doc_ids": self.doc_ids}, f)
        print(f"Saved BM25 index to {save_path}")

    def load_index(self, load_path: str = INDEX_PATH):
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]

    def retrieve(self, claim: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            raise AssertionError("Call build_index() or load_index() first")
        tokens = self._tokenize(claim)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]

    def retrieve_batch(self, claims: List[str], top_k: int = 5) -> List[List[Tuple[int, float]]]:
        if self.bm25 is None:
            raise AssertionError("Call build_index() or load_index() first")
        return [self.retrieve(claim, top_k=top_k) for claim in claims]


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, k: int = 60):
        self.dense = dense
        self.bm25 = bm25
        self.k = k

    def _rrf_fuse(self, dense_results, bm25_results, top_k: int):

        rrf_scores: Dict[int, float] = {}

        for rank, (doc_id, _) in enumerate(dense_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1/ (self.k + rank + 1)

        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1/ (self.k + rank + 1)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def retrieve(self, claim: str, top_k: int = 5, candidate_k: int = 20) -> List[Tuple[int, float]]:
        dense_results = self.dense.retrieve(claim, top_k=candidate_k)
        bm25_results = self.bm25.retrieve(claim, top_k=candidate_k)
        return self._rrf_fuse(dense_results, bm25_results, top_k=top_k)

    def retrieve_batch(self, claims: List[str], top_k: int = 5, candidate_k: int = 20) -> List[List[Tuple[int, float]]]:
        dense_batch = self.dense.retrieve_batch(claims, top_k=candidate_k)
        bm25_batch = self.bm25.retrieve_batch(claims, top_k=candidate_k)
        fused = []
        for dense_results, bm25_results in zip(dense_batch, bm25_batch):
            fused.append(self._rrf_fuse(dense_results, bm25_results, top_k=top_k))
        return fused
