# !pip install -qU git+https://github.com/naver/splade.git
# !pip install torch

import torch
from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print("==========\n"+
          "WARNING: You are not running on GPU so this may be slow.\n")

sparse_model_id = 'naver/splade-cocondenser-ensembledistil'

sparse_model = Splade(sparse_model_id, agg='max')
sparse_model.to(device) 
sparse_model.eval()

tokenizer = AutoTokenizer.from_pretrained(sparse_model_id)
tokens = tokenizer(data[0]['context'], return_tensors='pt')

with torch.no_grad():
    sparse_emb = sparse_model(
        d_kwargs=tokens.to(device)
    )['d_rep'].squeeze()
print(sparse_emb.shape)
indices = sparse_emb.nonzero().squeeze().cpu().tolist()
print(len(indices))
values = sparse_emb[indices].cpu().tolist()
sparse = {'indices': indices, 'values': values}