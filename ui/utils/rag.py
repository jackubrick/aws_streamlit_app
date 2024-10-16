from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from utils import constants as const
from langchain_openai import AzureOpenAI
from dotenv import load_dotenv
import os


# Load .env file
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

retriever = const.DB.as_retriever(k=5) 

qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | const.RAG_PROMPT
    | llm_00
    | StrOutputParser()
)