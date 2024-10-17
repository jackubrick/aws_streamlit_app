from dotenv import load_dotenv
import boto3
import json

# # Load AWS environment variables from the .env file
# load_dotenv()
# client = boto3.client('s3')
# response = client.list_buckets()
# print(response['Buckets'])


bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

text = "London is the capital of the United Kingdom and is nicknamed THE BIG SMOKE. It is the most populous city in England and the United Kingdom, with a metropolitan population of 15.8 million."

request_body = json.dumps({
    "inputText": text
})

model_id = 'amazon.titan-embed-text-v2:0'  

response = bedrock_client.invoke_model(
    modelId=model_id,
    body=request_body,
    accept='application/json',
    contentType='application/json'
)

response_body = json.loads(response['body'].read())
embeddings = response_body.get('embedding')

print(embeddings)