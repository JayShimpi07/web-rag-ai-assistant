import streamlit as st
import os
from dotenv import load_dotenv

from core.loader import load_documents
from core.splitter import split_documents
from core.embeddings import get_embeddings
from core.vectorstore import build_vectorstore
from core.chain import process_query
from core.memory import init_memory, add_to_memory, get_memory, reset_memory

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Web Knowledge AI",
    layout="wide",
    page_icon="🤖"
)

# ---------------- GLOBAL STYLING ----------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg,#6366f1,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    margin-bottom: 20px;
}

.metric-card {
    background-color: #0f172a;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.chat-container {
    background-color: #111827;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #1f2937;
}

.stButton>button {
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

load_dotenv()
init_memory()

# ---------------- SESSION INIT ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {"docs": 0, "chunks": 0, "avg_chunk_len": 0}

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🤖 Web Knowledge AI Assistant</div>', unsafe_allow_html=True)
st.caption("Multi-Source Retrieval Augmented AI System")
st.divider()

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📂 Ingestion", "💬 Chat", "📊 Analytics"])

# ==========================================================
# 📂 TAB 1 — INGESTION
# ==========================================================
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Add Knowledge Sources")

        url_input = st.text_input("Enter URL")

        uploaded_files = st.file_uploader(
            "Upload Files",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True
        )

        text_input = st.text_area("Direct Text Input")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Processing")

        if st.button("🚀 Process Information", use_container_width=True):

            if not (url_input or uploaded_files or text_input):
                st.warning("Please provide at least one source.")
            else:
                with st.spinner("Ingesting knowledge..."):

                    all_docs = []

                    # URL
                    if url_input:
                        all_docs.extend(load_documents(url_input, "url"))

                    # Files
                    if uploaded_files:
                        for file in uploaded_files:
                            file_type = file.name.split(".")[-1].lower()
                            all_docs.extend(load_documents(file, file_type))

                    # Direct Text
                    if text_input:
                        all_docs.extend(load_documents(text_input, "text"))

                    if all_docs:
                        chunks = split_documents(all_docs)
                        embeddings = get_embeddings()
                        st.session_state.vectorstore = build_vectorstore(chunks, embeddings)

                        avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)

                        st.session_state.doc_stats = {
                            "docs": len(all_docs),
                            "chunks": len(chunks),
                            "avg_chunk_len": round(avg_len, 2)
                        }

                        st.success("Knowledge Base Updated Successfully!")
                    else:
                        st.error("No content extracted.")

        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# 💬 TAB 2 — CHAT
# ==========================================================
with tab2:

    st.subheader("Chat with Your Knowledge Base")

    # Status
    if st.session_state.vectorstore:
        st.success("🟢 Knowledge Base Active")
    else:
        st.warning("🟡 Please process data first")

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display history
    for chat in get_memory():
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["ai"])

    query = st.chat_input("Ask something...")

    if query:

        with st.chat_message("user"):
            st.write(query)

        if not st.session_state.vectorstore:
            with st.chat_message("assistant"):
                st.error("No knowledge base available.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):

                    result = process_query(st.session_state.vectorstore, query)
                    answer = result["answer"]
                    sources = result["sources"]

                    # Streaming effect
                    placeholder = st.empty()
                    full_text = ""
                    for word in answer.split():
                        full_text += word + " "
                        placeholder.markdown(full_text)

                    add_to_memory(query, answer)

                    if sources:
                        with st.expander("📚 Sources & Similarity Scores"):
                            for idx, source in enumerate(sources):
                                st.markdown(f"**Source {idx+1}** (Score: {source['score']:.4f})")
                                st.markdown(f"*{source['metadata']}*")
                                st.text(source["content"][:300] + "...")
                                st.divider()

    st.markdown('</div>', unsafe_allow_html=True)

    # Reset Button
    if st.button("🧹 Reset Chat Memory"):
        reset_memory()
        st.rerun()

# ==========================================================
# 📊 TAB 3 — ANALYTICS
# ==========================================================
with tab3:

    if not st.session_state.vectorstore:
        st.info("No knowledge base created yet.")
    else:
        stats = st.session_state.doc_stats

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📄 Documents", stats["docs"])
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🧩 Chunks", stats["chunks"])
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📏 Avg Length", f"{stats['avg_chunk_len']} chars")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Configuration")
        st.json({
            "Embedding Model": "all-MiniLM-L6-v2",
            "Chunk Size": 1000,
            "Chunk Overlap": 200
        })

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with LangChain • FAISS • Groq Llama 3.1 • Streamlit")
