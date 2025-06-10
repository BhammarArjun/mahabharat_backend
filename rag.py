import asyncio
from groq import AsyncGroq
from bot import ContextualVectorDB  # Ensure this is accessible

# Initialize the DB once
contextual_db = ContextualVectorDB(name="my_contextual_db")

async def answer_from_context(user_query: str):
    # Step 1: Retrieve relevant context
    context_text = contextual_db.search(user_query, k=10)

    # Step 3: Use Groq LLM with the contextual prompt
    client = AsyncGroq()

    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"""
You are a Mahābhārata expert assistant.

- Always use the provided context to answer the user's question.
- If the context doesn't include the answer, clearly say you cannot find it in the context.
- If the user's question is unrelated to the Mahābhārata, politely say you only handle Mahābhārata-related queries.
- Do not invent or hallucinate information not supported by the context.
- Never break character or deviate from this responsibility.
""",
            },
            {
                "role": "user",
                "content": f"""current question: {user_query} \n\n context: {context_text}""",
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )

    return (chat_completion.choices[0].message.content, context_text)
