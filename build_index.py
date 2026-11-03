import os
import json
import faiss
from sentence_transformers import SentenceTransformer


# ==============================
# 1. Load KPR knowledge files
# ==============================

DATA_FOLDER = "data"

documents = []

for filename in os.listdir(DATA_FOLDER):

    if filename.endswith(".txt"):

        filepath = os.path.join(DATA_FOLDER, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        documents.append({
            "filename": filename,
            "text": text
        })


# ==============================
# 2. Split documents into chunks
# ==============================

chunks = []

for doc in documents:

    words = doc["text"].split()

    chunk_size = 100
    overlap = 20

    for i in range(0, len(words), chunk_size - overlap):

        chunk_words = words[i:i + chunk_size]

        if chunk_words:

            chunks.append({
                "filename": doc["filename"],
                "text": " ".join(chunk_words)
            })


# ==============================
# 3. Create embeddings
# ==============================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(texts)


# ==============================
# 4. Create FAISS index
# ==============================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# ==============================
# 5. Create models folder
# ==============================

os.makedirs("models", exist_ok=True)


# ==============================
# 6. Save FAISS index
# ==============================

faiss.write_index(index, "models/kai.index")


# ==============================
# 7. Save chunk information
# ==============================

with open("models/chunks.json", "w", encoding="utf-8") as file:

    json.dump(
        chunks,
        file,
        ensure_ascii=False,
        indent=2
    )


# ==============================
# 8. Display results
# ==============================

print("\n========== KAI RAG INDEX ==========")

print("Files loaded:", len(documents))

print("Total chunks:", len(chunks))

print("Embedding shape:", embeddings.shape)

print("Vectors stored in FAISS:", index.ntotal)

print("FAISS index saved: models/kai.index")

print("Chunk data saved: models/chunks.json")

print("\nFirst chunk:")
print(chunks[0]["text"])

print("\n===================================")
print("KAI RAG index built successfully!")