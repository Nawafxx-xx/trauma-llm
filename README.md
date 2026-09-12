# Grounded

**A context-aware trauma support assistant powered by RAG, conversational memory, and large language models.**

Grounded is an AI-powered conversational application designed to explore how Retrieval-Augmented Generation (RAG) and conversation memory can improve supportive, context-aware dialogue.

Rather than relying only on the general knowledge of an LLM, Grounded retrieves relevant examples from a dataset of therapist-client conversations and uses them as background context while generating a response.

The retrieved conversations are treated as **reference examples only** and are never assumed to describe the current user.

---

## Architecture

```text
Frontend
   ↓
FastAPI
   ↓
Chat Engine
   ├── Conversation Memory
   ├── Prompt Engineering
   └── RAG
        ↓
   FAISS Vector Store
        ↓
   MiniLM Embeddings
   ↓
Groq API
   ↓
Qwen
```

---

## Features

* Retrieval-Augmented Generation using therapy conversation data
* Semantic search with FAISS
* Local sentence-transformer embeddings
* Conversation-level retrieval with neighboring context chunks
* Session-based conversational memory
* Isolation between different chat sessions
* Prompt engineering to separate retrieved examples from the current user's history
* FastAPI backend
* Groq-hosted Qwen inference
* Structured request and response validation using Pydantic
* Designed for integration with a Flutter chat frontend

---

## RAG Strategy

Instead of retrieving unrelated individual messages, Grounded preserves conversational context.

Each therapy conversation is divided into overlapping three-turn sequences.

Every chunk contains metadata:

```text
conversation_id
chunk_index
```

When a user sends a message:

1. The message is converted into an embedding.
2. FAISS searches for the most semantically relevant conversation chunk.
3. The system identifies the original conversation.
4. Neighboring chunks from the **same conversation** are retrieved.
5. The resulting context is supplied to the LLM as background guidance.

This helps prevent the model from combining unrelated patient stories into one artificial conversation.

```text
User Query
    ↓
Semantic Search
    ↓
Best Matching Chunk
    ↓
Conversation ID + Chunk Index
    ↓
Previous + Current + Next Chunk
    ↓
RAG Context
```

---

## Conversation Memory

Each conversation receives a unique `session_id`.

```json
{
  "message": "I have been feeling anxious lately.",
  "session_id": "chat-001"
}
```

Grounded maintains separate histories for each session.

This allows the model to understand follow-up messages while preventing conversation history from leaking between users or sessions.

During development, conversation history is stored using LangChain's in-memory chat history.

Persistent storage can later be introduced using Redis or a database.

---

## Tech Stack

### Backend

* Python
* FastAPI
* Pydantic
* Uvicorn

### LLM & Orchestration

* LangChain
* Groq
* Qwen

### RAG

* FAISS
* Sentence Transformers
* `all-MiniLM-L6-v2`

### Planned Frontend

* Flutter

### Deployment

* Render — FastAPI backend
* Groq — LLM inference
* Flutter Web — planned public interface

---

## Project Structure

```text
Grounded-ai/
│
├── api.py
├── chat_engine.py
├── rag.py
├── indexing.py
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

### `api.py`

Defines the FastAPI application and API endpoints.

Responsibilities include:

* receiving chat requests
* validating request data
* forwarding messages to the chat engine
* returning model responses

### `chat_engine.py`

Coordinates the conversational pipeline.

Responsibilities include:

* conversation history
* prompt construction
* retrieved RAG context
* LLM inference
* storing assistant and user messages

### `rag.py`

Handles runtime retrieval.

Responsibilities include:

* loading the FAISS index
* embedding incoming queries
* semantic search
* identifying the best conversation
* retrieving neighboring chunks
* constructing the final RAG context

### `indexing.py`

Handles offline dataset preprocessing.

Responsibilities include:

* loading therapy conversations
* creating overlapping dialogue chunks
* generating embeddings
* attaching conversation metadata
* building the FAISS vector index

Indexing is performed separately so the production application does not need to repeatedly process the original dataset.

---

## Running Locally

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```text
GROQ_API_KEY=your_api_key
```

Start the FastAPI server:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Example Request

### POST `/chat`

```json
{
  "message": "I'm scared that talking about what happened will make me lose control.",
  "session_id": "demo-user-01"
}
```

Example response:

```json
{
  "response": "..."
}
```

---

## Safety & Design Considerations

Grounded is an experimental AI project and is **not a replacement for professional mental-health care, diagnosis, or emergency services**.

The system is intentionally designed so that:

* retrieved therapy conversations are treated only as background examples
* retrieved patient details are not assumed to describe the current user
* conversation histories remain isolated by session
* internal retrieved documents should not be exposed verbatim to users
* generated responses should avoid making clinical diagnoses
* safety-sensitive conversations require additional safeguards before production use

---

## Current Development Status

Implemented:

* FastAPI backend
* Groq / Qwen integration
* LangChain orchestration
* conversation memory
* FAISS vector search
* local MiniLM embeddings
* conversation-aware RAG
* neighboring-chunk retrieval
* session isolation
* prompt-based separation between retrieved patients and current users

In progress:

* retrieval quality evaluation
* similarity thresholds for irrelevant queries
* safety-layer improvements
* production deployment

Planned:

* Flutter chat interface
* persistent conversation storage
* streaming responses
* improved retrieval evaluation
* public web deployment

---

## Purpose

Grounded was built as an applied AI engineering project to explore the complete lifecycle of a modern LLM application:

```text
Data Processing
→ Embeddings
→ Vector Search
→ RAG
→ Prompt Engineering
→ Conversation Memory
→ LLM Inference
→ Backend APIs
→ Deployment
→ Mobile/Web Integration
```

The project focuses not only on calling an LLM API, but on designing the surrounding retrieval, memory, backend, and deployment architecture required for a complete conversational AI system.

---

## Disclaimer

This project is intended for educational, research, and portfolio purposes.

It does not provide medical advice, psychotherapy, diagnosis, or emergency mental-health services.
