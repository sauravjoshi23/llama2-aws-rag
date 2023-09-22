# !pip install -qU \
#     sagemaker==2.173.0 \
#     pinecone-client==2.2.1 \
#     ipywidgets==7.0.0

import json
import boto3

def create_payload(question):
    prompts = [question]
    payloads = []
    for prompt in prompts:
        payloads.append(
            {
                "inputs": prompt, 
                "parameters": {"max_new_tokens": 10, "top_p": 0.9, "temperature": 0.3, "return_full_text": False},
            }
        )
    return payloads[0]

endpoint_name = 'jumpstart-dft-meta-textgeneration-llama-2-7b'

def query_llama2_7b_endpoint(payload):
    client = boto3.client("sagemaker-runtime")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
        CustomAttributes="accept_eula=true",
    )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    return response[0]['generation']

def encode(text: str):
    dense_embeddings = encoder.predict({"inputs": [text]})
    dense_vec = np.mean(np.array(dense_embeddings), axis=1)
    dense_vec = dense_vec.tolist()
    input_ids = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        sparse_vec = sparse_model(
            d_kwargs=input_ids.to(device)
        )['d_rep'].squeeze()
    indices = sparse_vec.nonzero().squeeze().cpu().tolist()
    values = sparse_vec[indices].cpu().tolist()
    sparse_dict = {"indices": indices, "values": values}
    return dense_vec, sparse_dict

def rag_query(question: str) -> str:
    dense, sparse = encode(question)
    xc = index.query(
        vector=dense,
        sparse_vector=sparse,
        top_k=2, 
        include_metadata=True
    )
    context_str = xc['matches'][0]['metadata']['context'] + ' ' + xc['matches'][1]['metadata']['context']
    text_input = prompt_template.replace("{context}", context_str).replace("{question}", question)
    payload = create_payload(text_input)
    generated_text = query_llama2_7b_endpoint(payload)
    return generated_text

prompt_template = """Answer the following QUESTION based on the CONTEXT given. 

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

rag_query("Which lace plant produces perforations in its leaves through PCD?")