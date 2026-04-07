from typing import List

from langchain_core.runnables import RunnableLambda


class HybridRetriever:
    """
    Combines BM25 (keyword) and vector search (semantic).
    Uses Reciprocal Rank Fusion to merge results intelligently.
    """
    
    def __init__(self, bm25_retriever, vector_retriever, weights=[0.4, 0.6]):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.bm25_weight = weights[0]  # 40% keyword
        self.vector_weight = weights[1]  # 60% semantic
    
    def _retrieve(self, query: str) -> List:
        """Internal retrieval method using RRF."""
        # Get results from both retrievers
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)
        
        # Reciprocal Rank Fusion (RRF) scoring
        doc_scores = {}
        
        # Score BM25 results
        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.page_content
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            # RRF formula: score = weight / (rank + k), where k=60 is standard
            doc_scores[doc_id]["score"] += self.bm25_weight / (rank + 60)
        
        # Score vector results
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.page_content
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            doc_scores[doc_id]["score"] += self.vector_weight / (rank + 60)
        
        # Sort by combined score and return documents
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]
    
    def invoke(self, query: str) -> List:
        """Public method for retrieval."""
        return self._retrieve(query)
    
    def as_runnable(self):
        """Convert to Runnable for use in LCEL chains."""
        return RunnableLambda(self._retrieve)

print("✅ Custom Hybrid Retriever defined")