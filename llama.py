from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)

class ChatRequest(BaseModel):
    query: str
    temperature: float = 0.7

class ChatResponseModel(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponseModel)
async def chat_with_model(request: ChatRequest):
    try:
        # Interact with the ollama chat model
        response = chat(
            model='llama3.2-vision',
            messages=[
                {
                    'role': 'user',
                    'content': request.query,
                },
            ],
            options={
                'temperature': request.temperature
            }
        )

        # Return the response content
        return {"response": response['message']['content']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
