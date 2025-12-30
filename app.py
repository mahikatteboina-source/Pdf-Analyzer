import streamlit as st
from pypdf import PdfReader
import math
from collections import Counter
import re

# ------------------ Text Utilities ------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def tf(text):
    words = clean_text(text)
    return Counter(words)

def cosine_similarity(v1, v2):
    intersection = set(v1.keys()) & set(v2.keys())
    numerator = sum(v1[x] * v2[x] for x in intersection)
    denom1 = math.sqrt(sum(v**2 for v in v1.values()))
    denom2 = math.sqrt(sum(v**2 for v in v2.values()))
    return numerator / (denom1 * denom2) if denom1 and denom2 else 0.0

def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# ------------------ UI ------------------

st.set_page_config(page_title="PDF Analyzer", layout="centered")
st.title("📄 Smart PDF Analyzer (Fast & Stable)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    full_text = " ".join([page.extract_text() or "" for page in reader.pages])

    chunks = chunk_text(full_text)
    chunk_vectors = [tf(chunk) for chunk in chunks]

    st.success(f"Processed {len(chunks)} text chunks.")

    query = st.text_input("Ask a question about the document")

    if query:
        query_vec = tf(query)
        scores = [cosine_similarity(query_vec, v) for v in chunk_vectors]

        best_index = scores.index(max(scores))
        best_answer = chunks[best_index]

        st.subheader("📌 Best Match")
        st.write(best_answer)
