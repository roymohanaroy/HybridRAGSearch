import streamlit as st
import requests

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="Hybrid RAG Comparator",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/compare"  # change this

# -------------------------------
# TITLE
# -------------------------------
st.title("🔍 Hybrid RAG System Comparator")
st.markdown("Compare **Basic RAG vs Enhanced RAG** in real-time")

# -------------------------------
# INPUT
# -------------------------------
query = st.text_input("💬 Enter your question")

# Example prompts
examples = [
    "How does multi-head attention work?",
    "What is self-attention complexity?",
    "Why is Transformer better than RNNs?",
    "How many parameters does the base model have?"
]

st.sidebar.header("💡 Example Questions")
for ex in examples:
    if st.sidebar.button(ex):
        query = ex

# -------------------------------
# CALL BACKEND
# -------------------------------
if st.button("🚀 Compare") and query:

    with st.spinner("Running RAG systems..."):

        try:
            response = requests.post(
                API_URL,
                json={"query": query},
                timeout=60
            )

            data = response.json()

        except Exception as e:
            st.error(f"Backend error: {e}")
            st.stop()

    # -------------------------------
    # RESULTS UI
    # -------------------------------
    st.divider()

    col1, col2 = st.columns(2)

    # -------------------------------
    # BASIC RAG
    # -------------------------------
    with col1:
        st.subheader("🔵 Basic RAG")

        st.write(data["basic"]["answer"])

        st.markdown("### 📊 Stats")
        st.metric("Chunks Used", data["basic"]["num_sources"])
        st.metric("Word Count", data["basic"]["word_count"])

    # -------------------------------
    # ENHANCED RAG
    # -------------------------------
    with col2:
        st.subheader("🟢 Enhanced RAG")

        st.write(data["enhanced"]["answer"])

        st.markdown("### 📊 Stats")
        st.metric("Retrieved Docs", data["enhanced"]["retrieved"])
        st.metric("After Rerank", data["enhanced"]["reranked"])
        st.metric("Word Count", data["enhanced"]["word_count"])

    # -------------------------------
    # INSIGHT PANEL
    # -------------------------------
    st.divider()
    st.subheader("📈 Comparison Insight")

    basic_len = data["basic"]["word_count"]
    enhanced_len = data["enhanced"]["word_count"]

    if enhanced_len > basic_len * 1.3:
        st.success("✅ Enhanced RAG provides more detailed answers")
    elif enhanced_len < basic_len:
        st.warning("⚠️ Basic RAG is more concise")
    else:
        st.info("ℹ️ Both systems are similar in verbosity")

    # -------------------------------
    # RAW JSON (debug mode)
    # -------------------------------
    with st.expander("🔍 Raw Response"):
        st.json(data)