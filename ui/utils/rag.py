from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import AzureOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import os

# LLM
load_dotenv()
azure_endpoint = os.getenv('AZURE_ENDPOINT')
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_api_version = os.getenv('OPENAI_API_VERSION')

llm_00 = AzureOpenAI(
    deployment_name = 'gpt-35-turbo-instruct',
    azure_endpoint= azure_endpoint,            
    openai_api_key = openai_api_key,           
    openai_api_version = openai_api_version,    
    temperature = 0.0
)

# VECTOR STORE
VECTOR_DB_CACHE = "vectorstore"
embedding_function = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'}, # Change to cuda to force using GPU
    encode_kwargs={'normalize_embeddings': True} # set True to compute cosine similarity
)
vectorstore = Chroma(persist_directory=VECTOR_DB_CACHE, embedding_function=embedding_function)
retriever = vectorstore.as_retriever(k=5) 

# PROMPT
prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
# prompt = hub.pull("rlm/rag-prompt") - RAG prompt from hwchase
prompt = ChatPromptTemplate.from_template(prompt_template)

# CHAIN
qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm_00
    | StrOutputParser()
)