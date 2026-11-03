import requests

# Read KPR information
with open("data/kpriet_info.txt", "r", encoding="utf-8") as file:
    kpr_info = file.read()

url = "http://localhost:11434/api/generate"

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    prompt = f"""
You are KAI, the AI assistant for KPR Institute of Engineering and Technology.

Use the following KPR information to answer the user's question.

KPR INFORMATION:
{kpr_info}

USER QUESTION:
{question}

Answer based on the provided KPR information. If the information is not available, say that you don't have that information.
"""

    data = {
        "model": "qwen2.5:3b",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=data)

    print("KAI:", response.json()["response"])
    