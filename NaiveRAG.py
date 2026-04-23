from typing import Any

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from typing import Dict
from typing import Any
from langsmith import traceable

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

class BasicRAG:
    """
    The RAG system that works in demos but fails with real users.
    Study this to learn what NOT to do.
    """
    
    def __init__(self, pdf_path: str):
        print("file path-------------*********************************------------------------",pdf_path)
        self.pdf_path = pdf_path
        self.vectorstore = None
        self.chain = None
        
    def setup(self):
        """Setup the naive pipeline - looks simple, works poorly."""
        print("🔵 Building Basic RAG (the problematic one)...")
        
        # Step 1: Load the PDF
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        if not documents:
           raise ValueError(f"❌ No documents loaded from {self.pdf_path}")
        print(f"   Loaded {len(documents)} pages from PDF")
        
        # Step 2: Split it (badly)
        # Problem: No overlap = lost context at boundaries
        # Problem: Fixed 1000 chars = cuts sentences randomly
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,      # ❌ BIG MISTAKE #1: No context preservation
            separator="\n"
        )
        chunks = text_splitter.split_documents(documents)
        print(f"   Split into {len(chunks)} chunks (watch them break later...)")
        
        # Step 3: Embed and store (only vector search, no BM25)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="basic_rag",
            persist_directory="./basic_db"
        )
        print("   Stored embeddings in ChromaDB")
        
        # Step 4: Create chain using modern LCEL
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Simple prompt (too vague)
        prompt = PromptTemplate.from_template(
            "Answer the question based on the context:\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Build the chain using LCEL
        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("   QA chain ready")
        print("   ⚠️  Warning: This WILL disappoint you\n")
    
    @traceable   
    def query(self, question: str) -> Dict[str, Any]:
        """Ask a question, get a questionable answer."""
        # Get the answer
       
        answer = self.chain.invoke(question)
        
        # Also get source docs for comparison
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        source_docs = retriever.invoke(question)
        
        return {
            "answer": answer,
            "source_docs": source_docs,
            "num_sources": len(source_docs)
        }

print("✅ Basic RAG class defined")