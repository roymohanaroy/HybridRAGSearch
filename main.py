from fastapi import FastAPI
from pydantic import BaseModel

from EnhancedRAGSearch import EnhancedRAG
from NaiveRAG import BasicRAG



app = FastAPI()

# ✅ Load models ONCE at startup
basic_rag = BasicRAG("docs/1706.03762.pdf")
basic_rag.setup()

enhanced_rag = EnhancedRAG("docs/1706.03762.pdf")
enhanced_rag.setup()


# ✅ Request schema
class QueryRequest(BaseModel):
    query: str


# ✅ API endpoint
@app.post("/compare")
def compare(req: QueryRequest):
    question = req.query

    basic_result = basic_rag.query(question)
    enhanced_result = enhanced_rag.query(question)

    return {
        "question": question,
        "basic": {
            "answer": basic_result["answer"],
            "num_sources": basic_result["num_sources"],
            "word_count": len(basic_result["answer"].split())
        },
        "enhanced": {
            "answer": enhanced_result["answer"],
            "retrieved": enhanced_result["total_retrieved"],
            "reranked": enhanced_result["after_reranking"],
            "word_count": len(enhanced_result["answer"].split())
        }
    }