# !pip install -qU datasets

from datasets import load_dataset

pubmed = load_dataset(
    'pubmed_qa',
    'pqa_labeled',
    split='train'
)

print(pubmed[0]['pubid'], pubmed[0]['context'])

limit = 384

def chunker(contexts: list):
    chunks = []
    all_contexts = ' '.join(contexts).split('.')
    chunk = []
    for context in all_contexts:
        chunk.append(context)
        if len(chunk) >= 3 and len('.'.join(chunk)) > limit:
            chunks.append('.'.join(chunk).strip()+'.')
            chunk = chunk[-2:]
    if chunk is not None:
        chunks.append('.'.join(chunk))
    return chunks

chunks = chunker(pubmed[0]['context']['contexts'])

ids = []
for i in range(len(chunks)):
    ids.append(f"{pubmed[0]['pubid']}-{i}")

data = []
for record in pubmed:
    chunks = chunker(record['context']['contexts'])
    for i, context in enumerate(chunks):
        data.append({
            'id': f"{record['pubid']}-{i}",
            'context': context
        })

print(data[:2])