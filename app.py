from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from simple_revise_agent import query_revise

app = FastAPI()

# Add CORS middleware to allow requests from any origin with any method (you can customize as needed)
origins = ["*"]  # Allow requests from any origin
# Add CORS middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/reviseandretrieve/")
async def revise_and_retrieve(query: str):
    res = {"Agent Response": query_revise(query)}
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)