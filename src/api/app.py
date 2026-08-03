from fastapi import FastAPI
from pydantic import BaseModel

from src.llm.rag_pipeline import ask_rag


app = FastAPI(
    title="Enterprise RAG Document Q&A System",
    description="API for answering questions from enterprise documents using RAG and Llama 3.1",
    version="1.0"
)


class QuestionRequest(BaseModel):

    question: str



@app.get("/")
def home():

    return {
        "message": "Enterprise RAG API is running"
    }



@app.post("/ask")
def ask_question(request: QuestionRequest):

    answer = ask_rag(request.question)

    return {
        "question": request.question,
        "answer": answer
    }