import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

# NEW imports (work on Streamlit Cloud)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
st.title("📘 RAG Chatbot – Milestone 1 Helper")
st.write("Ask any question about the MS1 Checklist PDF!")


# ----------------------------------------------------------
# Load PDF
# ----------------------------------------------------------
def load_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        c = page.extract_text()
        if c:
            text += c + "\n"
    return text

pdf_text = load_text("ms1.pdf")


# ----------------------------------------------------------
# Chunking
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
retriever = db.as_retriever()


# ----------------------------------------------------------
# LLM (Gemma)
# ----------------------------------------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

llm = HuggingFaceHub(
    repo_id="google/gemma-2-2b-it",
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={"temperature": 0.1, "max_new_tokens": 350}
)


# ----------------------------------------------------------
# Build Retrieval Chain (NEW LC API)
# ----------------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Use ONLY the provided context to answer.

    Context:
    {context}

    Question:
    {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)


# ----------------------------------------------------------
# Streamlit interaction
# ----------------------------------------------------------
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip():
        response = rag_chain.invoke({"input": query})

        st.write("### ✅ Answer:")
        st.write(response["answer"])

        with st.expander("📄 Retrieved Chunks"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("---")
