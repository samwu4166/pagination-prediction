from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from ..dependencies import get_token_header
import uuid
from urllib.parse import unquote
import sys
import os
from datetime import datetime
from bson.objectid import ObjectId

#import page CRUDs
from ..database import (
    add_page,
    delete_page,
    retrieve_page,
    retrieve_pages,
    update_page,
)
from ..models.page import (
    ErrorResponseModel,
    ResponseModel,
    PageSchema,
    RequestPageSchema,
    UpdatePageModel,
)


sys.path.insert(0, os.path.abspath(r"."))
# print(os.path.abspath(r"."))

from autopager.autopager import get_shared_autopager
from autopager.preprocessing import generate_page_component

shared_autopager = get_shared_autopager()

router = APIRouter(

    prefix="/autopager",

    tags=["autopager"],

    # dependencies=[Depends(get_token_header)],

    responses={404: {"description": "Not found"}},

)



@router.get("/")
async def read_pages():
    pages = await retrieve_pages()
    if pages:
        return ResponseModel(pages, "Pages data retrieved.")
    return ResponseModel(pages, "Empty list returned.")



@router.get("/{tid}")
async def read_page(tid: str):
    page = await retrieve_page(tid)
    if page:
        return ResponseModel(page, "Page data retrieved.")
    return ErrorResponseModel("An error occured", 404, "Page data doesn't exist.")


@router.post("/")
async def predict_page(page: RequestPageSchema = Body(...)):
    print("CURRENT_CRF_MODEL: ",shared_autopager.DEFAULT_CRF_PATH)
    _uid = ObjectId()
    page_item = jsonable_encoder(page)
    uncoded_url = unquote(page_item['url'])
    # print(f"Get request: {uncoded_url}")
    autopager = get_shared_autopager()
    page_component = generate_page_component(uncoded_url)
    result_urls = autopager.urls(page_component["html"], uncoded_url, direct=True, prev=False, next=False)
    current_time = datetime.now()
    result_component = {"tid": _uid, "url": uncoded_url, "urls": result_urls, "created_time": current_time}
    ### add result to Mongo
    new_page = await add_page_data(result_component)
    return ResponseModel(new_page, "Page predicted and saved successfully.")

async def add_page_data(page: PageSchema):
    new_page = await add_page(page)
    return new_page

@router.put("/{tid}")
async def update_page_data(tid: str, req: UpdatePageModel = Body(...)):
    req = {k: v for k, v in req.dict().items() if v is not None}
    updated_page = await update_page(tid, req)
    if updated_page:
        return ResponseModel(
            "Page with ID: {} update is successful".format(tid),
            "Page updated successfully",
        )
    return ErrorResponseModel(
        "An error occurred",
        404,
        "There was an error updating the page data.",
    )

@router.delete("/{tid}", response_description="Page data deleted from the database")
async def delete_page_data(tid: str):
    deleted_page = await delete_page(tid)
    if deleted_page:
        return ResponseModel(
            "Page with ID: {} removed".format(tid), "Page deleted successfully"
        )
    return ErrorResponseModel(
        "An error occurred", 404, "Page with tid {0} doesn't exist".format(tid)
    )