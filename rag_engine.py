import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough   
from langchain_groq import ChatGroq  
from dotenv import load_dotenv


# Charger les variables du fichier .env
load_dotenv()  

# Récupérer la clé
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# CONFIGURATION
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


HUGGINGFACE_MODEL = "HuggingFaceH4/zephyr-7b-beta"

RAG_PROMPT_TEMPLATE = """
<|system|>
Tu es un Expert en Recrutement Technique. Analyse le CV fourni pour répondre à la question.
Si l'information n'est pas dans le CV, dis "Non mentionné".
CONTEXTE :
{context}</s>
<|user|>
{question}</s>
<|assistant|>
"""

def process_pdf_and_create_vector_db(uploaded_file):
    if uploaded_file is None: return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    finally:
        if os.path.exists(tmp_file_path): os.remove(tmp_file_path)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vector_store, groq_api_key):
    llm = ChatGroq(
        temperature=0.2,
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain