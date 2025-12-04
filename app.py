import streamlit as st
from pypdf import PdfReader

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
st.title("📘 RAG Chatbot – Milestone 1 Helper")
st.write("Ask any question about the MS1 Checklist PDF. The answer is generated using FAISS-based retrieval + Gemma LLM.")

# ----------------------------------------------------------
# 1. Load PDF
# ----------------------------------------------------------
def load_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

pdf_text = load_text("ms1.pdf")

# ----------------------------------------------------------
# 2. Split into chunks
# ----------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)
chunks = splitter.split_text(pdf_text)

# ----------------------------------------------------------
# 3. Embeddings + FAISS
# ----------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_texts(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# ----------------------------------------------------------
# 4. LLM (Gemma 2B IT)
# ----------------------------------------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

llm = HuggingFaceHub(
    repo_id="google/gemma-2-2b-it",
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={"temperature": 0.2, "max_new_tokens": 350}
)

# ----------------------------------------------------------
# 5. Retrieval QA Chain
# ----------------------------------------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# ----------------------------------------------------------
# 6. UI
# ----------------------------------------------------------
query = st.text_input("Enter your question about Milestone 1:")

if st.button("Get Answer"):
    if query.strip():
        result = qa(query)

        st.write("### ✅ Answer:")
        st.write(result["result"])

        st.write("---")
        with st.expander("📄 Retrieved Source Chunks"):
            for i, doc in enumerate(result["source_documents"], 1):
                st.write(f"**Chunk {i}:**")
                st.write(doc.page_content)
                st.write("---")
