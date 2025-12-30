import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    # MiniLM is lightweight, fast, and accurate for semantic search
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- APP UI ----------------
st.set_page_config("Smart PDF Analyzer", layout="centered")
st.title("📄 Intelligent PDF Analyzer (Semantic Search)")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    reader = PdfReader(uploaded)
    full_text = " ".join(page.extract_text() or "" for page in reader.pages)

    # Chunk text for better retrieval granularity
    def chunk_text(text, size=350, overlap=80):
        words = text.split()
        chunks = []
        for i in range(0, len(words), size - overlap):
            chunks.append(" ".join(words[i:i + size]))
        return chunks

    chunks = chunk_text(full_text)

    # Precompute embeddings for all chunks
    embeddings = model.encode(chunks, convert_to_tensor=True)

    st.success(f"Processed {len(chunks)} document sections.")

    query = st.text_input("Ask a question about the document")

    if query:
        q_emb = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, embeddings)[0]

        # Get top 3 relevant sections
        top_k = min(3, len(chunks))
        top_results = scores.topk(top_k)

        st.subheader("📌 Best Answer (Summarized from Document)")
        combined_answer = "\n\n".join([chunks[int(idx)] for idx in top_results.indices])
        st.write(combined_answer)

        # Show scores for transparency
        st.subheader("🔎 Match Scores")
        for idx, score in zip(top_results.indices, top_results.values):
            st.write(f"Section {int(idx)+1} — Score: {score:.4f}")
