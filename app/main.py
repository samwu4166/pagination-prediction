from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .dependencies import get_query_token, get_token_header
from .routers import autopager

app = FastAPI(
    title="Pagination Prediction",
    description="This is a RESTAPI project of pagination prediction.",
    # dependencies=[Depends(get_query_token)],
    version="0.0.1",
)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(autopager.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}