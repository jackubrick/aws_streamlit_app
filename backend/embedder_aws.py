from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_chroma import Chroma
import boto3
import json

# Initialize Bedrock client
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

# Custom Bedrock embedding class to match LangChain's expectations
class BedrockEmbedding:
    def __init__(self, model_id='amazon.titan-embed-text-v2:0'):
        self.model_id = model_id

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            # Prepare request body for each text
            body = json.dumps({
                "inputText": text,
                "dimensions": 1024,
                "normalize": True
            })
            # Invoke the Bedrock model
            response = bedrock_client.invoke_model(
                modelId=self.model_id,
                body=body,
                accept='application/json',
                contentType='application/json'
            )
            # Parse the response and extract embeddings
            response_body = json.loads(response['body'].read())
            embeddings.append(response_body.get('embedding'))
        return embeddings

# Replace the old EMBEDDING_FUNCTION with the new BedrockEmbedding class
EMBEDDING_FUNCTION = BedrockEmbedding()

VECTOR_DB_CACHE = "vectorstore"
vectorstore = Chroma(persist_directory=VECTOR_DB_CACHE, embedding_function=EMBEDDING_FUNCTION)

# Define text splitter
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

def store_vector(document):
    texts = TEXT_SPLITTER.split_text(document)
    embeddings = EMBEDDING_FUNCTION.embed_documents(texts)  # Use the new embed_documents method
    vectorstore.add_texts(texts=texts, embeddings=embeddings, persist_directory=VECTOR_DB_CACHE)

# Load documents
loader = DirectoryLoader("ec2_data", glob="*.docx", loader_cls=UnstructuredWordDocumentLoader)
documents = loader.load()

# Store vectors for each document
for document in documents:
    store_vector(document.page_content)
