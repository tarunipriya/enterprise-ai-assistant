from groq import Groq
from app.config import settings

client = Groq(api_key=settings.GROQ_API_KEY)


def generate_response(question: str, context: str):

    prompt = f"""
You are an AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply:

"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content