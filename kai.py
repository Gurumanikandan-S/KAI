import requests
import faiss
import json
from sentence_transformers import SentenceTransformer


# ==============================
# 1. Load RAG components
# ==============================

print("Loading KAI...")

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("models/kai.index")

with open("models/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("KAI is ready!")


# ==============================
# 2. Ollama settings
# ==============================

OLLAMA_URL = "http://localhost:11434/api/generate"

LLM_MODEL = "qwen2.5:3b"


# ==============================
# 3. KAI chatbot
# ==============================

while True:

    question = input("\nYou: ")

    if question.lower().strip() == "exit":
        print("KAI: Goodbye!")
        break


    # ==============================
    # 4. Handle basic greetings
    # ==============================

    greetings = {
        "hi": "Hi! 👋 I'm KAI, your KPR Institute AI assistant. How can I help you?",
        "hello": "Hello! 👋 How can I help you with KPR information?",
        "hey": "Hey! 👋 What would you like to know about KPR?",
        "good morning": "Good morning! ☀️ How can I help you?",
        "good afternoon": "Good afternoon! How can I help you?",
        "good evening": "Good evening! How can I help you?"
    }

    user_input = question.lower().strip()

    if user_input in greetings:
        print("\nKAI:", greetings[user_input])
        continue


    # --------------------------
    # Convert question to vector
    # --------------------------

    question_embedding = model.encode([question])


    # --------------------------
    # Search FAISS
    # --------------------------

    k = 3

    distances, indices = index.search(
        question_embedding,
        k
    )


    # --------------------------
    # Get relevant chunks
    # --------------------------

    retrieved_text = ""

    for i, index_number in enumerate(indices[0]):

        retrieved_text += (
            f"\nSource: {chunks[index_number]['filename']}\n"
            f"{chunks[index_number]['text']}\n"
        )


    # --------------------------
    # Create prompt for Qwen
    # --------------------------

    prompt = f"""
You are KAI, the AI assistant for KPR Institute of Engineering and Technology.

Answer the user's question using ONLY the information provided in the context below.

If the answer cannot be found in the context, say:
"I don't have that information in my KPR knowledge base."

Do not invent information.

CONTEXT:
{retrieved_text}

USER QUESTION:
{question}

Give a clear and concise answer.
"""


    # --------------------------
    # Send to Qwen
    # --------------------------

    data = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }


    response = requests.post(
        OLLAMA_URL,
        json=data
    )


    # --------------------------
    # Display answer
    # --------------------------

    if response.status_code == 200:

        answer = response.json()["response"]

        print("\nKAI:", answer)

    else:

        print("\nError:", response.status_code)