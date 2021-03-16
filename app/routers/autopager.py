from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_token_header

import uuid

from urllib.parse import unquote

import sys

sys.path.insert(0, '../../..')

from autopager.autopager import get_shared_autopager

shared_autopager = get_shared_autopager()

router = APIRouter(

    prefix="/autopager",

    tags=["autopager"],

    # dependencies=[Depends(get_token_header)],

    responses={404: {"description": "Not found"}},

)



fake_items_db = {"test_url1": {"page": ["url1","url2"], "next": ["url1"]}, "test_url2": {"page": ["url1","url2"], "next": ["url1", "url2"]}}



@router.get("/")

async def read_items():
    print("CURRENT_CRF_MODEL: ",shared_autopager.DEFAULT_CRF_PATH)
    return fake_items_db



@router.get("/{url_id}")

async def read_item(url_id: str):
    if url_id not in fake_items_db:
        raise HTTPException(status_code=404, detail="URL not found")
    return {"result": fake_items_db[url_id], "url_id": url_id}


@router.post("/{encoded_url}")
async def predict_item(encoded_url: str):
    _uid = uuid.uuid1()
    uncoded_url = unquote(encoded_url)
    print("Get post action: ", uncoded_url)
    return {"uuid": _uid, "result": {"page": ["url1","url2"], "next": ["url1"]}}
