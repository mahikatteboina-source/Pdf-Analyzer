import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI ----------------
st.set_page_config(page_title="PDF Analyzer", layout="wide")
st.title("📄 PDF Analyzer (No Tokenizers, Streamlit Safe)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# ---------------- PDF Processing ----------------
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    documents = text_splitter.split_documents(pages)

    texts = [doc.page_content for doc in documents]

    # Vectorize using TF-IDF (NO tokenizers)
    vectorizer = TfidfVectorizer(stop_words="english")
    embeddings = vectorizer.fit_transform(texts)

    st.success("✅ Document processed successfully!")

    # ---------------- QUERY ----------------
    query = st.text_input("Ask a question about the PDF")

    if query:
        query_vec = vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, embeddings)[0]
        best_idx = similarity.argmax()

        st.subheader("📌 Best Answer")
        st.write(texts[best_idx])
