A Hybrid Retrieval-Augmented Generation (RAG) system that combines the strengths of multiple retrieval strategies to deliver more accurate, context-aware, and reliable LLM responses.

This project compares and integrates:

🧠 Naive RAG (Basic Retrieval)
⚡ Enhanced Hybrid RAG (Advanced Retrieval + Ranking + Context Optimization)

Built with FastAPI backend and supports Streamlit UI for interactive testing and comparison.

📌 Features
🔍 Dual RAG System Comparison
Basic RAG vs Enhanced Hybrid RAG
📊 Side-by-side response evaluation
⚡ FastAPI-powered backend for inference APIs
🧩 Modular architecture (easy to extend retrieval logic)
📄 Supports document-based question answering
🧠 LLM-powered response generation (Groq / OpenAI / etc.)
📈 Designed for experimentation and benchmarking
🏗️ Architecture
User Query
     ↓
FastAPI Backend
     ↓
┌──────────────────────┐
│  Basic RAG Pipeline  │
└──────────────────────┘
            vs
┌──────────────────────┐
│ Enhanced Hybrid RAG  │
│ (Vector + Ranking +  │
│  Context Optimization│
└──────────────────────┘
     ↓
LLM (Response Generation)
     ↓
Final Answer / Comparison Output
⚙️ Tech Stack
Python 3.10+
FastAPI – backend API layer
Streamlit – frontend UI (optional)
LangChain / LLM APIs – reasoning layer
Vector Database / Embeddings (depending on setup)
Groq / OpenAI models (configurable LLM backend)
🚀 Getting Started
1️⃣ Clone the repository
git clone https://github.com/roymohanaroy/HybridRAGSearch.git
cd HybridRAGSearch
2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
▶️ Run FastAPI Backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

If deploying on Render:

uvicorn main:app --host 0.0.0.0 --port $PORT
🧪 API Endpoints
🔹 Compare RAG Systems
POST /compare
Request Body:
{
  "basic_rag": "BasicRAG config or input",
  "enhanced_rag": "EnhancedRAG config or input",
  "questions": ["What is RAG?", "Explain hybrid search"]
}
Response:
Side-by-side outputs from both systems
Comparative insights
🖥️ Streamlit UI (Optional)

If Streamlit frontend is included:

streamlit run app.py

Features:

Enter query
Compare Basic vs Hybrid RAG responses
Visual output comparison
📊 Use Cases
RAG benchmarking
LLM response quality comparison
Research on retrieval strategies
AI agent development experiments
Document QA systems
🧠 Why Hybrid RAG?

Traditional RAG systems suffer from:

❌ Weak retrieval relevance
❌ Poor ranking of context chunks
❌ Hallucination in answers

This hybrid approach improves:

✅ Retrieval quality
✅ Context relevance
✅ Answer accuracy
✅ Robustness across queries
📌 Future Improvements
Add reranking models (Cross-Encoders)
Add BM25 + Vector hybrid search
Add evaluation metrics dashboard
Support for multi-document reasoning
Add LangGraph-based agent pipeline
👨‍💻 Author

Mohana Roy
GitHub: @roymohanaroy

⭐ If you like this project

Give it a ⭐ on GitHub and contribute improvements!
