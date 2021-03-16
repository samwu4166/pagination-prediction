from fastapi import Header, HTTPException

async def get_token_header(API_token: str = Header(...)):
    if API_token != "fake-API-token":
        raise HTTPException(status_code=400, detail="API-Token header invalid")

async def get_query_token(token: str):
    if token != "jessica":
        raise HTTPException(status_code=400, detail="No Jessica token provided")
