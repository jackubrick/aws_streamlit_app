from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

EMBEDDING_FUNCTION = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'}, # Change to cuda to force using GPU
    encode_kwargs={'normalize_embeddings': True} # set True to compute cosine similarity
)

VECTOR_DB_CACHE = "vectorstore"
vectorstore = Chroma(persist_directory=VECTOR_DB_CACHE, embedding_function=EMBEDDING_FUNCTION)

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 100,
    length_function = len,
    is_separator_regex = False,
)

def store_vector(document):
    texts = TEXT_SPLITTER.split_text(document)
    vectorstore.add_texts(texts=texts,
                    embedding=EMBEDDING_FUNCTION,
                    persist_directory=VECTOR_DB_CACHE) # Persist the vector store

# LOAD DOCX FILES IN MEMORY:
loader = DirectoryLoader("data/S3_bucket", glob="*.docx", loader_cls=UnstructuredWordDocumentLoader)
documents = loader.load() # Load all .docx files from the S3 folder

for document in documents:
    store_vector(document.page_content)