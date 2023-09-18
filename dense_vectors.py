# !pip install -qU \
#     sagemaker==2.173.0 \
#     pinecone-client==2.2.1 \
#     ipywidgets==7.0.0

hub_config = {
    'HF_MODEL_ID': 'sentence-transformers/all-MiniLM-L6-v2', 
    'HF_TASK': 'feature-extraction'
}

huggingface_model = HuggingFaceModel(
    env=hub_config,
    role=role,
    transformers_version="4.6",
    pytorch_version="1.7", 
    py_version="py36", 
)

encoder = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.large",
    endpoint_name="minilm-demo"
)

out = encoder.predict({"inputs": ["some text here", "some more text goes here too"]})
print(len(out[0][0]))