import streamlit as st
from pypdf import PdfReader
import re
import math
from collections import Counter

# ---------------- TEXT PROCESSING ----------------

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def chunk_text(text, size=350, overlap=80):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i + size]))
    return chunks

def term_frequency(words):
    return Counter(words)

def cosine_similarity(v1, v2):
    common = set(v1) & set(v2)
    numerator = sum(v1[x] * v2[x] for x in common)
    denom = (sum(v**2 for v in v1.values()) ** 0.5) * (sum(v**2 for v in v2.values()) ** 0.5)
    return numerator / denom if denom else 0

# ---------------- APP UI ----------------

st.set_page_config("Smart PDF Analyzer", layout="centered")
st.title("📄 Intelligent PDF Analyzer")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    reader = PdfReader(uploaded)
    full_text = " ".join(page.extract_text() or "" for page in reader.pages)

    chunks = chunk_text(full_text)
    chunk_vectors = [term_frequency(clean(chunk)) for chunk in chunks]

    st.success(f"Processed {len(chunks)} document sections.")

    query = st.text_input("Ask a question about the document")

    if query:
        query_vec = term_frequency(clean(query))

        # Score ALL chunks
        scores = [cosine_similarity(query_vec, v) for v in chunk_vectors]

        # Get top 3 relevant sections
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

        st.subheader("📌 Best Answer (Summarized from Document)")

        combined_answer = "\n\n".join([chunks[i] for i in top_indices])
        st.write(combined_answer)
