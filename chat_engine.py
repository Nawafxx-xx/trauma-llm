
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rag import retrieve_context
from langchain_groq import ChatGroq
load_dotenv()


chat_histories = {}
def get_chat_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]


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
def generate_response(message: str, session_id: str) -> str:
    chat_history = get_chat_history(session_id)
    
    context = retrieve_context(message)
    formatted_prompt = prompt.invoke({
    "context": context,
    "history":  chat_history.messages,
    "message": message
     })
    
    response = llm.invoke(formatted_prompt)
    answer = str(response.content)
    with_without=f" Rag: {answer}. \n without {llm.invoke(message).content}"
    chat_history.add_message(HumanMessage(content=message))
    chat_history.add_message(AIMessage(content=answer))
    return with_without
    