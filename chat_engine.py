
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rag import retrieve_context
from langchain_groq import ChatGroq
load_dotenv()






llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.7,
    reasoning_effort="none",
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a trauma-informed supportive assistant.
Never mention retrieved context, RAG, source conversations, examples from other patients, or the retrieval process to the user.
Use retrieved conversations silently as background guidance.
Never imply that the user provided the retrieved context.
Never treat details from retrieved conversations as facts about the current user.
Use the retrieved therapy context below only as background guidance.
Do not assume the retrieved context describes the current user.
Do not copy therapist responses verbatim.

Retrieved context:
{context}
"""),

    MessagesPlaceholder(variable_name="history"),

    ("human", "{message}")
])

def convert_history(history):
    messages = []

    for item in history:
        if item["role"] == "user":
            messages.append(
                HumanMessage(content=item["content"])
            )

        elif item["role"] == "assistant":
            messages.append(
                AIMessage(content=item["content"])
            )

    return messages

def generate_response(
    message: str,
    history: list,
    name: str
) -> str:

    langchain_history = convert_history(history)

    context = retrieve_context(message)

    formatted_prompt = prompt.invoke({
        "name": name,
        "context": context,
        "history": langchain_history,
        "message": message
    })

    response = llm.invoke(formatted_prompt)

    return response.content
    