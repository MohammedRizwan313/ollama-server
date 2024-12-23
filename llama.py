from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from ollama import chat

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

class ChatResponseModel(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponseModel)
async def chat_with_model(request: ChatRequest):
    try:
        # Interact with the ollama chat model
        response = chat(model='llama3.2', messages=[
            {
                'role': 'user',
                'content': request.query,
            },
        ])

        # Return the response content
        return {"response": response['message']['content']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
