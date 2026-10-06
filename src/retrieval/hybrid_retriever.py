import json
import os
import re
import hashlib
from typing import List, Dict, Any, Optional
from functools import lru_cache

# Optional heavy dependencies with resilient fallbacks
try:
    import numpy as np
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    HEAVY_DEPS_AVAILABLE = True
except ImportError:
    HEAVY_DEPS_AVAILABLE = False
    np = None
    faiss = None
    BM25Okapi = None
    SentenceTransformer = None


def _tokenize(text: str) -> List[str]:
    """Clean alphanumeric tokenization without punctuation artifacts."""
    return re.findall(r'[a-zA-Z0-9_\-]+', text.lower())


class SimpleBM25Fallback:
    """Lightweight pure-python fallback for sparse search if rank_bm25 is not installed."""
    def __init__(self, tokenized_corpus: List[List[str]]):
        self.corpus = tokenized_corpus

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        query_set = set(query_tokens)
        scores = []
        for doc in self.corpus:
            doc_set = set(doc)
            overlap = len(query_set.intersection(doc_set))
            scores.append(float(overlap))
        return scores


class HybridClinicalRetriever:
    """
    High-Performance Hybrid Clinical Retriever.
    
    Optimizations:
    1. Persistent FAISS Indexing: Caches vector index on disk to eliminate re-encoding corpus on startup.
    2. LRU Query Caching: Instantaneous responses for repeated patient queries.
    3. Resilient Architecture: Operates smoothly even if heavy vector dependencies are not yet installed.
    4. Enhanced Tokenization: Punctuation stripping for higher retrieval accuracy.
    """
    def __init__(self, guidelines_path: str = "data/guidelines.json", model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        self.guidelines_path = os.path.abspath(guidelines_path)
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.dirname(self.guidelines_path)
        
        # Load corpus
        with open(self.guidelines_path, "r", encoding="utf-8") as f:
            self.corpus: List[Dict[str, Any]] = json.load(f)
            
        self.docs = [doc["text"] for doc in self.corpus]
        self.tokenized_corpus = [_tokenize(doc) for doc in self.docs]
        
        # In-memory query cache
        self._query_cache: Dict[str, str] = {}
        
        # 1. Initialize Sparse BM25
        if BM25Okapi is not None:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = SimpleBM25Fallback(self.tokenized_corpus)
            
        # 2. Initialize Dense FAISS & SentenceTransformer
        self.encoder = None
        self.faiss_index = None
        
        if HEAVY_DEPS_AVAILABLE:
            self._init_dense_index()

    def _get_corpus_hash(self) -> str:
        """Computes MD5 hash of corpus to validate index cache freshness."""
        hasher = hashlib.md5()
        for doc in self.docs:
            hasher.update(doc.encode('utf-8'))
        return hasher.hexdigest()

    def _init_dense_index(self):
        """Initializes or loads persistent FAISS index from disk."""
        index_file = os.path.join(self.cache_dir, "guidelines_faiss.index")
        meta_file = os.path.join(self.cache_dir, "guidelines_faiss_meta.json")
        current_hash = self._get_corpus_hash()
        
        index_loaded = False
        if os.path.exists(index_file) and os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("hash") == current_hash and meta.get("model") == self.model_name:
                    self.faiss_index = faiss.read_index(index_file)
                    self.encoder = SentenceTransformer(self.model_name)
                    index_loaded = True
            except Exception:
                index_loaded = False

        if not index_loaded:
            # Build index from scratch
            self.encoder = SentenceTransformer(self.model_name)
            # Direct normalization in PyTorch
            embeddings = self.encoder.encode(self.docs, convert_to_numpy=True, normalize_embeddings=True)
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings)
            
            # Persist to disk
            try:
                faiss.write_index(self.faiss_index, index_file)
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump({"hash": current_hash, "model": self.model_name, "count": len(self.docs)}, f)
            except Exception:
                pass

    def dense_search(self, query: str, top_k: int = 3) -> List[int]:
        if not self.faiss_index or not self.encoder:
            return list(range(min(top_k, len(self.docs))))
            
        query_vector = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        _, indices = self.faiss_index.search(query_vector, top_k)
        return indices[0].tolist()

    def sparse_search(self, query: str, top_k: int = 3) -> List[int]:
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        if np is not None and isinstance(scores, np.ndarray):
            top_indices = np.argsort(scores)[::-1][:top_k].tolist()
        else:
            # Pure python sort
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in indexed_scores[:top_k]]
            
        return top_indices

    def reciprocal_rank_fusion(self, dense_ranks: List[int], sparse_ranks: List[int], k: int = 60) -> List[int]:
        """Standard RRF scoring: Score(d) = SUM(1 / (k + rank))."""
        rrf_scores: Dict[int, float] = {}
        
        for rank, doc_idx in enumerate(dense_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + (rank + 1)))
            
        for rank, doc_idx in enumerate(sparse_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + (rank + 1)))
            
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_idx for doc_idx, _ in sorted_docs]

    def retrieve_context(self, query: str, top_k: int = 2) -> str:
        """Retrieves and formats top guideline chunks with memoization caching."""
        cache_key = f"{query.strip().lower()}__k{top_k}"
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        if HEAVY_DEPS_AVAILABLE and self.faiss_index:
            dense_results = self.dense_search(query, top_k=top_k * 2)
            sparse_results = self.sparse_search(query, top_k=top_k * 2)
            fused_indices = self.reciprocal_rank_fusion(dense_results, sparse_results, k=60)
            selected_indices = fused_indices[:top_k]
        else:
            selected_indices = self.sparse_search(query, top_k=top_k)

        retrieved_chunks = []
        for idx in selected_indices:
            doc = self.corpus[idx]
            retrieved_chunks.append(f"[{doc['source']} - {doc['topic']}]\n{doc['text']}")
            
        result = "\n\n".join(retrieved_chunks)
        self._query_cache[cache_key] = result
        return result