from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chat_engine import generate_response


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://grounded-web-lovat.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    client_id: str
    name: str
    message: str
    history: list[HistoryMessage] = []


class ChatResponse(BaseModel):
    response: str


@app.get("/")
def home():
    return {
        "message": "Grounded API is running"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    history = [
        {
            "role": item.role,
            "content": item.content
        }
        for item in request.history
    ]

    answer = generate_response(
        message=request.message,
        history=history,
        name=request.name
    )

    return {
        "response": answer
    }