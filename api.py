from pydantic import BaseModel
from fastapi import FastAPI
from chat_engine import generate_response

app = FastAPI()
@app.get("/")
def home():
    return {"message": "Trauma LLM API is running"}

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer = generate_response(request.message,request.session_id)
    return {"response": answer}
    

