import faiss
import json
from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("models/kai.index")

# Load saved chunks
with open("models/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)


# Ask a question
question = input("Enter your question: ")


# Convert question into an embedding
question_embedding = model.encode([question])


# Search FAISS
k = 3

distances, indices = index.search(question_embedding, k)


# Display retrieved information
print("\n========== RETRIEVED INFORMATION ==========")

for i, index_number in enumerate(indices[0]):

    print(f"\nResult {i + 1}")
    print("File:", chunks[index_number]["filename"])
    print("Distance:", distances[0][i])
    print("Text:")
    print(chunks[index_number]["text"])

print("\n===========================================")