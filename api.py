from fastapi import FastAPI
from pydantic import BaseModel
import requests
import faiss
import json
from sentence_transformers import SentenceTransformer


# ==============================
# KAI API
# ==============================

app = FastAPI(title="KAI - KPR AI")


# ==============================
# Load RAG components
# ==============================

print("Loading KAI...")

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("models/kai.index")

with open("models/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("KAI is ready!")


# ==============================
# Ollama settings
# ==============================

OLLAMA_URL = "http://localhost:11434/api/generate"

LLM_MODEL = "qwen2.5:3b"


# ==============================
# Request format
# ==============================

class Question(BaseModel):
    question: str


# ==============================
# Greeting handling
# ==============================

greetings = {
    "hi": "Hi! 👋 I'm KAI, your KPR Institute AI assistant. How can I help you?",
    "hello": "Hello! 👋 How can I help you with KPR information?",
    "hey": "Hey! 👋 What would you like to know about KPR?",
    "good morning": "Good morning! ☀️ How can I help you?",
    "good afternoon": "Good afternoon! How can I help you?",
    "good evening": "Good evening! How can I help you?"
}


# ==============================
# Ask KAI
# ==============================

@app.post("/ask")
def ask_kai(data: Question):

    question = data.question.strip()

    if not question:
        return {
            "answer": "Please enter a question."
        }


    # Check greeting

    user_input = question.lower()

    if user_input in greetings:

        return {
            "answer": greetings[user_input]
        }


    # ==============================
    # Convert question to vector
    # ==============================

    question_embedding = model.encode([question])


    # ==============================
    # Search FAISS
    # ==============================

    k = 3

    distances, indices = index.search(
        question_embedding,
        k
    )


    # ==============================
    # Get relevant information
    # ==============================

    retrieved_text = ""

    for index_number in indices[0]:

        retrieved_text += (
            f"\nSource: {chunks[index_number]['filename']}\n"
            f"{chunks[index_number]['text']}\n"
        )


    # ==============================
    # Create prompt
    # ==============================

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


    # ==============================
    # Send to Ollama
    # ==============================

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code == 200:

            answer = response.json()["response"]

            return {
                "answer": answer
            }

        else:

            return {
                "answer": "KAI could not connect to the AI model."
            }

    except requests.exceptions.ConnectionError:

        return {
            "answer": "Ollama is not running. Please start Ollama first."
        }