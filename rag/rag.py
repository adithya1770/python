import faiss
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import spacy
from sentence_transformers import SentenceTransformer

nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('all-MiniLM-L6-v2')
gpt2_model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
 
# chunking the files
with open('green day.txt', 'r') as file_content:
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

# embedding the file into vectors
embed_file = model.encode(chunk)

#indexing the embeds using faiss
embed_file = np.array(embed_file)
dimension = embed_file.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embed_file)
faiss.write_index(index, 'embed_index.faiss')

# getting user prompt
query = input("Enter user prompt:")
query_embedding = model.encode([query])

# checking the indices/vectors of input query with the text file query
k = 3  
distances, indices = index.search(query_embedding, k)
relevant_chunks = [chunk[i] for i in indices[0]]
augmented_input = query + " " + " ".join(relevant_chunks)


# generating response using local model
inputs = tokenizer.encode(augmented_input, return_tensors='pt')
output = model.generate(inputs, max_length=150, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
