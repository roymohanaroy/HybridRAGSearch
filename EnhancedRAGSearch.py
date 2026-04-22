from typing import List

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict
from typing import Any
from sentence_transformers import CrossEncoder
from HybridRetriever import HybridRetriever

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

class EnhancedRAG:
    """
    The RAG system that survived production.
    Built after learning all the painful lessons.
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.vectorstore = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.chain = None
        self.chunks = None
        
    def setup(self):
        """Setup production RAG - complex but actually works."""
        print("🟢 Building Enhanced RAG (the one that works)...")
        # Step 6: Cross-Encoder for re-ranking
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # fast & effective
        print("   Cross-Encoder ready for semantic re-ranking")
        
        # Step 1: Load PDF (same as before)
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        if not documents:
          raise ValueError(f"❌ No documents loaded from {self.pdf_path}")
        print(f"   Loaded {len(documents)} pages")
        
        # Step 2: Smart chunking with context preservation
        # This was THE game changer
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,           # Smaller = more precise retrieval
            chunk_overlap=200,        # ✅ CRUCIAL: Preserves context!
            separators=["\n\n", "\n", ". ", " ", ""],  # Respects structure
            length_function=len,
        )
        self.chunks = text_splitter.split_documents(documents)
        
        # Add metadata for tracking and debugging
        for i, chunk in enumerate(self.chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['char_count'] = len(chunk.page_content)
        
        print(f"   Created {len(self.chunks)} overlapping chunks")
        
        # Step 3: Build TWO retrievers (not one!)
        
        # Retriever 1: Semantic search (vector)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            collection_name="enhanced_rag",
            persist_directory="./enhanced_db"
        )
        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 6}  # Get more, filter later
        )
        
        # Retriever 2: Keyword search (BM25)
        # This catches exact term matches that vector search misses
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = 6
        
        # Combine them using our custom hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            vector_retriever=vector_retriever,
            weights=[0.4, 0.6]  # 40% keyword, 60% semantic
        )
        print("   Built hybrid retriever (Keyword + Semantic)")
        
        # Step 4: Enhanced prompt that actually guides the model
        template = """You are a helpful AI assistant answering questions about a research paper.
Use ONLY the context below to answer. If the context doesn't contain the answer, say so.
IMPORTANT: Provide complete, self-contained answers. Include specific facts, numbers, 
and formulas directly in your response. DO NOT just reference tables or sections 
(like "see Table 3" or "refer to section 2.1"). Instead, extract and include the 
actual information in your answer so it's useful even without the original document.
Context: {context}
Question: {question}
Detailed Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Step 5: Build the chain using modern LCEL
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Modern chain with hybrid retrieval
        self.chain = (
            {"context": self.hybrid_retriever.as_runnable() | format_docs, 
             "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("   QA chain built with enhanced prompt")
        print("   ✅ Ready to actually help users\n")
    
    def rerank_documents(self, query: str, documents: List, top_k: int = 4):
            """
            Rerank retrieved documents using a Cross-Encoder for semantic relevance.
            """
            if not documents:
                return []

            # Prepare pairs for cross-encoder
            pairs = [(query, doc.page_content) for doc in documents]

            # Get semantic relevance scores
            scores = self.cross_encoder.predict(pairs)

            # Sort documents by score descending
            ranked_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]

            return ranked_docs[:top_k]
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query with automatic reranking - much better results."""
        # Get answer
        answer = self.chain.invoke(question)
        
        # Get source documents
        source_documents = self.hybrid_retriever.invoke(question)
        
        # Rerank for quality
        reranked_docs = self.rerank_documents(
            question,
            source_documents,
            top_k=4
        )
        
        return {
            "answer": answer,
            "source_docs": reranked_docs,
            "total_retrieved": len(source_documents),
            "after_reranking": len(reranked_docs)
        }

print("✅ Enhanced RAG class defined")