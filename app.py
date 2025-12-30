import streamlit as st
import numpy as np
from pypdf import PdfReader
from collections import Counter
import math

def text_to_vector(text):
    words = text.lower().split()
    return Counter(words)

def cosine_sim(a, b):
    intersection = set(a) & set(b)
    num = sum(a[x] * b[x] for x in intersection)
    denom = math.sqrt(sum(v*v for v in a.values())) * math.sqrt(sum(v*v for v in b.values()))
    return num / denom if denom else 0

st.title("PDF Search (Streamlit Safe)")

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    reader = PdfReader(file)
    docs = [p.extract_text() for p in reader.pages if p.extract_text()]

    vectors = [text_to_vector(t) for t in docs]

    query = st.text_input("Ask a question")
    if query:
        q_vec = text_to_vector(query)
        scores = [cosine_sim(q_vec, v) for v in vectors]
        best = docs[scores.index(max(scores))]
        st.write(best)
