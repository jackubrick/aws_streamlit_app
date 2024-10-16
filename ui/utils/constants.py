from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma # UPDATE TO: from langchain_chroma import Chroma
# from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


MODEL_NAME = "BAAI/bge-small-en-v1.5"

EMBEDDING_FUNCTION = HuggingFaceBgeEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'}, # Change to cuda to force using GPU
    encode_kwargs={'normalize_embeddings': True} # set True to compute cosine similarity
)

VECTOR_DB_CACHE = "vectorstore"
DB = Chroma(persist_directory=VECTOR_DB_CACHE, embedding_function=EMBEDDING_FUNCTION)

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 100,
    length_function = len,
    is_separator_regex = False,
)

RAG_TEMPLATE = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
# prompt = hub.pull("rlm/rag-prompt")
RAG_PROMPT = ChatPromptTemplate.from_template(RAG_TEMPLATE)
