import streamlit as st
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from huggingface_hub import InferenceClient
from typing import Optional

# Manual RAG implementation using LangChain components
class CustomHFLLM:
    def __init__(self, hf_token):
        self.client = InferenceClient(model="google/gemma-2-2b-it", token=hf_token)
    
    def __call__(self, prompt):
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2
        )
        return response.choices[0].message["content"]

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def setup_rag(_hf_token, pdf_path="ms1.pdf"):
    # Extract text from PDF
    reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    # Create chunks using LangChain's text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    documents = splitter.create_documents([text])
    
    # Create FAISS vectorstore using LangChain
    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Setup custom LLM
    llm = CustomHFLLM(_hf_token)
    
    return retriever, llm

def query_rag(question, retriever, llm):
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(question)
    
    # Build context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Build prompt for RetrievalQA-style interaction
    prompt = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
    
    # Get answer from LLM
    answer = llm(prompt)
    
    return answer, docs

# Main app
st.title("RAG Chatbot")

# Get HF token from secrets or input
hf_token = st.secrets.get("HF_TOKEN", None) or st.text_input("HuggingFace Token:", type="password")

if hf_token:
    try:
        retriever, llm = setup_rag(hf_token)
        
        # Question input
        question = st.text_input("Ask a question:")
        
        if st.button("Ask") and question:
            with st.spinner("Thinking..."):
                answer, source_docs = query_rag(question, retriever, llm)
                
                st.write("**Answer:**")
                st.write(answer)
                
                st.write("**Sources:**")
                for i, doc in enumerate(source_docs, 1):
                    st.text_area(f"Chunk {i}", doc.page_content, height=100)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Full error details:", str(e))
else:
    st.info("Enter your HuggingFace token to start")
