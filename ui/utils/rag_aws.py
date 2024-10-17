from dotenv import load_dotenv
import boto3
import json
from langchain_chroma import Chroma

load_dotenv()

# Initialize Bedrock client
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

class BedrockEmbedding:
    def __init__(self, model_id='amazon.titan-embed-text-v2:0'):
        self.model_id = model_id
        self.client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    def embed_query(self, query):
        """Embeds a single query."""
        request_body = json.dumps({
            "inputText": query,
            "dimensions": 1024,
            "normalize": True
        })
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=request_body,
            accept='application/json',
            contentType='application/json'
        )
        response_body = json.loads(response['body'].read())
        return response_body.get('embedding')

VECTOR_DB_CACHE = "vectorstore"
embedding_function = BedrockEmbedding()
vectorstore = Chroma(persist_directory=VECTOR_DB_CACHE, embedding_function=embedding_function)

retriever = vectorstore.as_retriever(k=5)


def aws_chain(query):

    retrieved_docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in retrieved_docs])
    full_prompt = f"""YOU MUST USE THE INFORMATION IN THE CONTEXT to answer the question. If you don't know the answer, just say that you don't know.
    Context:\n{context}\n\nQuestion: {query}"""

    request_body = {
        "inputText": full_prompt,
        "textGenerationConfig": {
            "maxTokenCount": 500,
            "temperature": 0.0
        }
    }

    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    request_json = json.dumps(request_body)

    response = client.invoke_model(
        modelId="amazon.titan-text-lite-v1",
        body=request_json
    )

    response_body = json.loads(response['body'].read())
    output_text = response_body["results"][0]["outputText"]

    return output_text
