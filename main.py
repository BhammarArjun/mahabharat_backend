from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer_from_context

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ask/", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    answer, _ = await answer_from_context(request.question)
    return AnswerResponse(answer=answer)