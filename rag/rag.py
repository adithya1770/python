import faiss
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import spacy
from sentence_transformers import SentenceTransformer

nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('all-MiniLM-L6-v2')
gpt2_model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to('cuda')  # Move model to GPU

with open("green day.txt", 'r') as file:
    file_content = file.read()

token_doc = nlp(file_content)
chunk = []
current = []
count = 0

for token in token_doc:
    current.append(token.text)
    count += len(token.text) + 1 
    
    if count > 50:
        chunk.append(' '.join(current))
        current = []
        count = 0

if current:
    chunk.append(' '.join(current))

def encode_chunks(chunks, model):
    embeddings = model.encode(chunks)
    return np.array(embeddings).astype(np.float32)

encoded_chunks = encode_chunks(chunk, model)

dimension = encoded_chunks.shape[1]
index = faiss.IndexFlatL2(dimension)
gpu_res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
gpu_index.add(encoded_chunks)

def search_index(query, model, gpu_index):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype(np.float32)
    k = 3 
    distances, indices = gpu_index.search(query_embedding, k)
    return indices[0]  

query = input("Enter user input: ")
relevant_indices = search_index(query, model, gpu_index)
relevant_chunks = [chunk[i] for i in relevant_indices]

augmented_input = query + " " + " ".join(relevant_chunks)

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

response = generate_response(augmented_input, gpt2_model, tokenizer)
print("Response:", response)
