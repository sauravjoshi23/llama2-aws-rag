import pinecone
from creds import creds
from tqdm.auto import tqdm

def builder(records: list):
    ids = [x['id'] for x in records]
    contexts = [x['context'] for x in records]
    dense_contexts = {"inputs": contexts}
    dense_embeddings = encoder.predict(dense_contexts)
    dense_vecs = np.mean(np.array(dense_embeddings), axis=1)
    dense_vecs = dense_vecs.tolist()
    input_ids = tokenizer(
        contexts, return_tensors='pt',
        padding=True, truncation=True
    )
    with torch.no_grad():
        sparse_vecs = sparse_model(
            d_kwargs=input_ids.to(device)
        )['d_rep'].squeeze()
    upserts = []
    for _id, dense_vec, sparse_vec, context in zip(ids, dense_vecs, sparse_vecs, contexts):
        indices = sparse_vec.nonzero().squeeze().cpu().tolist()
        values = sparse_vec[indices].cpu().tolist() 
        sparse_values = {
            "indices": indices,
            "values": values
        }
        metadata = {'context': context}
        upserts.append({
            'id': _id,
            'values': dense_vec,
            'sparse_values': sparse_values,
            'metadata': metadata
        })
    return upserts

pinecone.init(
    api_key=creds['PINECONE_API_KEY'], 
    environment=creds['PINECONE_ENV']  
)
index_name = 'pubmed-splade'
pinecone.create_index(
    index_name,
    dimension=384,
    metric="dotproduct"
)
index = pinecone.Index(index_name)
batch_size = 2
for i in tqdm(range(0, 1000, batch_size)):
    index.upsert(builder(data[i:i+batch_size]))