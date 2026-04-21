import streamlit as st
from EnhancedRAGSearch import EnhancedRAG
from NaiveRAG import BasicRAG

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Hybrid RAG Comparator",
    layout="wide"
)

st.title("🔍 RAG Comparison: Basic vs Enhanced")

# -------------------------------
# LOAD SYSTEMS (CACHE)
# -------------------------------
@st.cache_resource
def load_models():
    basic = BasicRAG("docs/1706.03762.pdf")
    basic.setup()

    enhanced = EnhancedRAG("docs/1706.03762.pdf")
    enhanced.setup()

    return basic, enhanced

basic_rag, enhanced_rag = load_models()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Settings")

example_questions = [
    "How does multi-head attention work in the Transformer?",
    "What is the time complexity of self-attention?",
    "Why is the Transformer better than RNNs for long sequences?",
    "How many parameters does the base model have?"
]

selected_question = st.sidebar.selectbox(
    "Try an example:",
    [""] + example_questions
)

# -------------------------------
# INPUT
# -------------------------------
query = st.text_input(
    "💬 Enter your question:",
    value=selected_question
)

# -------------------------------
# RUN QUERY
# -------------------------------
if st.button("🔎 Compare") and query:
    with st.spinner("Running both RAG systems..."):

        basic_result = basic_rag.query(query)
        enhanced_result = enhanced_rag.query(query)

    # -------------------------------
    # LAYOUT: SIDE BY SIDE
    # -------------------------------
    col1, col2 = st.columns(2)

    # -------------------------------
    # BASIC RAG
    # -------------------------------
    with col1:
        st.subheader("🔵 Basic RAG")

        st.write(basic_result["answer"])

        st.markdown("**Stats:**")
        st.write(f"- Chunks used: {basic_result['num_sources']}")
        st.write(f"- Answer length: {len(basic_result['answer'].split())} words")

    # -------------------------------
    # ENHANCED RAG
    # -------------------------------
    with col2:
        st.subheader("🟢 Enhanced RAG")

        st.write(enhanced_result["answer"])

        st.markdown("**Stats:**")
        st.write(f"- Retrieved: {enhanced_result['total_retrieved']}")
        st.write(f"- After reranking: {enhanced_result['after_reranking']}")
        st.write(f"- Answer length: {len(enhanced_result['answer'].split())} words")

    # -------------------------------
    # COMPARISON INSIGHT
    # -------------------------------
    st.divider()
    st.subheader("📊 Comparison Insight")

    basic_len = len(basic_result["answer"].split())
    enhanced_len = len(enhanced_result["answer"].split())

    if enhanced_len > basic_len * 1.3:
        st.success("✅ Enhanced RAG produced a more detailed answer")
    elif enhanced_len < basic_len:
        st.warning("⚠️ Basic RAG was more concise")
    else:
        st.info("ℹ️ Both answers are similar in length")
