import json
import os
from typing import List, Dict, Any
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class HybridClinicalRetriever:
    def __init__(self, guidelines_path: str = "data/guidelines.json", model_name: str = "all-MiniLM-L6-v2"):
        self.guidelines_path = os.path.abspath(guidelines_path)
        self.encoder = SentenceTransformer(model_name)
        
        # Load corpus
        with open(self.guidelines_path, "r", encoding="utf-8") as f:
            self.corpus: List[Dict[str, Any]] = json.load(f)
            
        self.docs = [doc["text"] for doc in self.corpus]
        
        # 1. Build Dense FAISS Index
        embeddings = self.encoder.encode(self.docs, convert_to_numpy=True)
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)
        
        # 2. Build Sparse BM25 Index
        tokenized_corpus = [doc.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def dense_search(self, query: str, top_k: int = 3) -> List[int]:
        query_vector = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        _, indices = self.faiss_index.search(query_vector, top_k)
        return indices[0].tolist()

    def sparse_search(self, query: str, top_k: int = 3) -> List[int]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices.tolist()

    def reciprocal_rank_fusion(self, dense_ranks: List[int], sparse_ranks: List[int], k: int = 60) -> List[int]:
        """
        Standard RRF scoring: Score(d) = SUM(1 / (k + rank))
        """
        rrf_scores = {}
        
        for rank, doc_idx in enumerate(dense_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + (rank + 1)))
            
        for rank, doc_idx in enumerate(sparse_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + (rank + 1)))
            
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_idx for doc_idx, _ in sorted_docs]

    def retrieve_context(self, query: str, top_k: int = 2) -> str:
        dense_results = self.dense_search(query, top_k=top_k * 2)
        sparse_results = self.sparse_search(query, top_k=top_k * 2)
        
        fused_indices = self.reciprocal_rank_fusion(dense_results, sparse_results, k=60)
        selected_indices = fused_indices[:top_k]
        
        retrieved_chunks = []
        for idx in selected_indices:
            doc = self.corpus[idx]
            retrieved_chunks.append(f"[{doc['source']} - {doc['topic']}]\n{doc['text']}")
            
        return "\n\n".join(retrieved_chunks)