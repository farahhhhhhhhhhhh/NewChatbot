import streamlit as st
from pypdf import PdfReader

# LangChain imports (works with langchain==0.1.20)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA


# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
st.title("📘 RAG Chatbot – Milestone 1 Helper")
st.write("Ask any question about the MS1 Checklist PDF. The answer is generated using FAISS + LangChain RetrievalQA + Gemma LLM.")


# ----------------------------------------------------------
# Load PDF
# ----------------------------------------------------------
def load_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

pdf_text = load_text("ms1.pdf")


# ----------------------------------------------------------
# Split into chunks
# ----------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)

chunks = splitter.split_text(pdf_text)


# ----------------------------------------------------------
# Embeddings + FAISS
# ----------------------------------------------------------
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_texts(chunks, emb)
retriever = db.as_retriever(search_kwargs={"k": 3})


# ----------------------------------------------------------
# LLM (Gemma 2B IT)
# ----------------------------------------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

llm = HuggingFaceHub(
    repo_id="google/gemma-2-2b-it",
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={"temperature": 0.2, "max_new_tokens": 350}
)


# ----------------------------------------------------------
# RetrievalQA (required by assignment)
# ----------------------------------------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# ----------------------------------------------------------
# UI Interaction
# ----------------------------------------------------------
query = st.text_input("Enter your question about Milestone 1:")

if st.button("Get Answer"):
    if query.strip():
        result = qa(query)

        st.write("### ✅ Answer:")
        st.write(result["result"])

        with st.expander("📄 Retrieved Chunks"):
            for i, doc in enumerate(result["source_documents"], 1):
                st.write(f"**Chunk {i}:**")
                st.write(doc.page_content)
                st.write("---")
