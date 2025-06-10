import asyncio
from groq import AsyncGroq
from bot import ContextualVectorDB  # Ensure this is accessible

# Initialize the DB once
contextual_db = ContextualVectorDB(name="my_contextual_db")

async def answer_from_context(user_query: str):
    # Step 1: Retrieve relevant context
    context_text = contextual_db.search(user_query, k=10)

    # Step 2: Use Groq LLM with the improved contextual prompt
    client = AsyncGroq()

    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"""
You are a Mahābhārata expert scholar. For this conversation, your entire memory and knowledge of the Mahābhārata is strictly and exclusively limited to the text provided in the 'context' section. You must act as if you know nothing else on the topic.

Your instructions are:
1.  Read the user's question and the provided context carefully.
2.  Synthesize a direct and confident answer to the question using only the information from your knowledge base (the context).
3.  **Crucially, do not refer to the source of your information.** Never use phrases like "According to the context," "Based on the text provided," or "The context states." You must answer as if you are recalling this information from your own expert memory.
4.  If your knowledge base does not contain the information to answer the question, state clearly and simply that you do not have knowledge about that specific detail. For example, say "I do not have information on that topic."
5.  If the user's question is unrelated to the Mahābhārata, politely state that you specialize only in Mahābhārata-related topics.
6.  Never invent or assume information. Your answers must be grounded exclusively in the knowledge provided.
7.  Maintain your persona as an expert scholar at all times.
""",
            },
            {
                "role": "user",
                "content": f"""current question: {user_query} \n\n context: {context_text}""",
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.5, # Temperature can be lowered slightly to 0.3-0.4 for more factual recall
        max_completion_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )
    print(chat_completion.choices[0].message.content)
    return (chat_completion.choices[0].message.content, context_text)