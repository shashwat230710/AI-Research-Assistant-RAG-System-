import streamlit as st
import tempfile

from utils import load_pdf, split_docs, create_vectorstore

from transformers import pipeline

# -------------------------------
# Load LLM (cached for performance)
# -------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="distilgpt2",   # lightweight + fast
    )

llm = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title("📄 AI Research Assistant (FREE - HuggingFace)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# -------------------------------
# Process PDF
# -------------------------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.success("✅ PDF uploaded successfully!")

    with st.spinner("Processing document..."):
        documents = load_pdf(file_path)
        chunks = split_docs(documents)
        vectorstore = create_vectorstore(chunks)

    st.success("✅ Document processed and indexed!")

    # -------------------------------
    # User Query
    # -------------------------------
    query = st.text_input("Ask a question from the document:")

    if query:
        with st.spinner("Generating answer..."):

            docs = vectorstore.similarity_search(query, k=3)

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
You are an AI research assistant.

Answer ONLY from the provided context.
If answer is not present, say "Not found in document".

Context:
{context}

Question:
{query}

Answer:
"""

            response = llm(
                prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7
            )

            answer = response[0]["generated_text"]

            # -------------------------------
            # Output
            # -------------------------------
            st.write("### 🤖 Answer")
            st.write(answer)

            # -------------------------------
            # Show Sources (IMPORTANT)
            # -------------------------------
            st.write("### 📚 Source Chunks")
            for i, doc in enumerate(docs):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc.page_content[:300] + "...")
                st.write("---")