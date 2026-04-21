from EnhancedRAGSearch import EnhancedRAG
from NaiveRAG import BasicRAG
from NaiveRAG import BasicRAG
from fastapi import FastAPI

def compare_systems(basic_rag, enhanced_rag, questions):
    """Run both systems and compare results."""
    
    print("\n" + "=" * 100)
    print(" " * 30 + "🔵 BASIC  vs  🟢 ENHANCED")
    print("=" * 100 + "\n")
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 100}")
        print(f"Question {i}: {question}")
        print(f"{'─' * 100}\n")
        
        # Get answers from both
        basic_result = basic_rag.query(question)
        enhanced_result = enhanced_rag.query(question)
        
        # Basic answer
        print("🔵 BASIC RAG:")
        print(f"{basic_result['answer']}\n")
        print(f"   └─ Used {basic_result['num_sources']} chunks\n")
        
        # Enhanced answer
        print("🟢 ENHANCED RAG:")
        print(f"{enhanced_result['answer']}\n")
        print(f"   └─ Retrieved {enhanced_result['total_retrieved']}, ")
        print(f"      used {enhanced_result['after_reranking']} after reranking\n")
        
        # Quick comparison
        basic_len = len(basic_result['answer'].split())
        enhanced_len = len(enhanced_result['answer'].split())
        
        print("📊 Quick Stats:")
        print(f"   Basic: {basic_len} words")
        print(f"   Enhanced: {enhanced_len} words")
        
        if enhanced_len > basic_len * 1.3:
            print("   → Enhanced gave more detailed answer ✅")
        
        print("\n")

print("✅ Comparison function defined")

if __name__ == "__main__":
    basic_rag = BasicRAG("docs/1706.03762.pdf")  # The Transformer paper
    basic_rag.setup()

    # Build the enhanced system
    enhanced_rag = EnhancedRAG("docs/1706.03762.pdf")
    enhanced_rag.setup()
    # Test questions about the Transformer paper
    questions = [
        "How does multi-head attention work in the Transformer?",
        "What is the time complexity of self-attention?",
        "Why is the Transformer better than RNNs for long sequences?",
        "How many parameters does the base model have?",
    ]
    # Run comparison (we already have basic_rag from earlier)
    compare_systems(basic_rag, enhanced_rag, questions)