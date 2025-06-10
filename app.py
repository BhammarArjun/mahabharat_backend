from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from rag import answer_from_context

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask/", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    answer, _ = await answer_from_context(request.question)
    return AnswerResponse(answer=answer)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)