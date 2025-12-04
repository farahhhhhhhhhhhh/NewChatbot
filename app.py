import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from huggingface_hub import InferenceClient
from typing import Optional, List, Any

# Custom LLM wrapper
class GemmaLLM(LLM):
    client: Any = None
    max_tokens: int = 500

    @property
    def _llm_type(self) -> str:
        return "gemma_hf"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
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
    
    # Create chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    documents = splitter.create_documents([text])
    
    # Create vectorstore
    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Setup LLM
    client = InferenceClient(model="google/gemma-2-2b-it", token=_hf_token)
    llm = GemmaLLM(client=client)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain

# Main app
st.title("RAG Chatbot")

# Get HF token from secrets or input
hf_token = st.secrets.get("HF_TOKEN", None) or st.text_input("HuggingFace Token:", type="password")

if hf_token:
    qa_chain = setup_rag(hf_token)
    
    # Question input
    question = st.text_input("Ask a question:")
    
    if st.button("Ask") and question:
        result = qa_chain({"query": question})
        
        st.write("**Answer:**")
        st.write(result['result'])
        
        st.write("**Sources:**")
        for i, doc in enumerate(result['source_documents'], 1):
            st.text_area(f"Chunk {i}", doc.page_content, height=100)
else:
    st.info("Enter your HuggingFace token to start")
