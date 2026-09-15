from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import Request
from chat_engine import generate_response



app = FastAPI()

@app.middleware("http")
async def debug_requests(request: Request, call_next):
    print("ORIGIN:", request.headers.get("origin"))
    print(
        "REQUEST METHOD:",
        request.headers.get("access-control-request-method")
    )
    print(
        "REQUEST HEADERS:",
        request.headers.get("access-control-request-headers")
    )

    response = await call_next(request)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://grounded-web-lovat.vercel.app",
        "https://grounded-poeajmkww-nawafalsharani-1520s-projects.vercel.app",
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