from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import json
import os

load_dotenv()
DATA_FOLDER='/Users/nawafalserhani/trauma-llm/ThousandVoicesOfTrauma/conversations'
json_files = [
    file for file in os.listdir(DATA_FOLDER)
    if file.endswith(".json")
]

def load_conversation(file_path: str, conversation_id: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sequences = data["three_turn_sequences"]
    documents = []
    for index, sequence in enumerate(sequences):
        chunk_text = "\n".join(sequence)
        
        document = Document(
        page_content=chunk_text,
            metadata={
                    "conversation_id": conversation_id,
                    "chunk_index": index
                          }
                        )
        documents.append(document)
    return documents

def load_all_conversations():
    all_documents = []
    for file in json_files:
        conversation_id = os.path.splitext(file)[0]
        file_path = os.path.join(DATA_FOLDER, file)
        try:
            documents = load_conversation(file_path, conversation_id)
            all_documents.extend(documents)
        except UnicodeDecodeError:
            print("Encoding problem in:", file)
            raise
            
    return all_documents


embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)

documents = load_all_conversations()
vector_store = FAISS.from_documents(
    documents,
    embeddings
)
vector_store.save_local("faiss_index")