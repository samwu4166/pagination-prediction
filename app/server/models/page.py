from typing import Optional
import datetime
from pydantic import BaseModel, Field



class RequestPageSchema(BaseModel):
    url: str = Field(...)

    class Config:
        schema_extra = {
            "example": {
                "url": "https://kktix.com/events",
            }
        }

class RequestPageFileSchema(BaseModel):
    url: str = Field(...)
    html: str = Field(...)

    class Config:
        schema_extra = {
            "example": {
                "url": "https://test_url.com",
                "html": "<html> ... </html",
            }
        }


class PageSchema(BaseModel):
    tid: str = Field(...)
    url: str = Field(...)
    urls: list = Field(...)
    created_time: datetime.datetime = Field(...)
    
    class Config:
        schema_extra = {
            "example": {
                "tid": "test_tid_123",
                "url": "https://kktix.com/events",
                "urls": [
                    "https://kktix.com/events", 
                    "https://kktix.com/events?page=2", 
                    "https://kktix.com/events?page=3", 
                    "https://kktix.com/events?page=4"
                ],
                "created_time": "example_time_zone"
            }
        }


class UpdatePageModel(BaseModel):
    tid: Optional[str]
    url: Optional[str]
    urls: Optional[list]

    class Config:
        schema_extra = {
            "example": {
                "tid": "test_update_tid_123",
                "url": "https://kktix.com/events",
                "urls": [
                    "https://kktix.com/events", 
                    "https://kktix.com/events?page=2",
                ],
            }
        }


def ResponseModel(data, message):
    return {
        "data": data,
        "code": 200,
        "message": message,
    }


def ErrorResponseModel(error, code, message):
    return {"error": error, "code": code, "message": message}