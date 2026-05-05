import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from prompts import SYSTEM_PROMPT

load_dotenv()

# ====================== CONFIG ======================
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama3-8b-8192"
MAX_TOKENS = 300
TEMPERATURE = 0.0

# ====================== QUESTIONS ======================
user_questions = [
    "How far away is the Louvre from the Eiffel Tower (in miles) if you are driving?",
    "Where is the Arc de Triomphe?",
    "What are the must-see artworks at the Louvre Museum?",
]

# ====================== CONVERSATION ======================
conversation: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

print("🗼 Paris Travel Assistant (powered by Groq + Llama-3-8B)\n")
print("Asking the 3 tourist questions...\n")

for question in user_questions:
    print(f"👤 User: {question}")
    
    conversation.append({"role": "user", "content": question})
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=conversation,
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_TOKENS
    )
    
    answer = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": answer})
    
    print(f"🤖 Assistant: {answer}\n")
    print("-" * 60 + "\n")