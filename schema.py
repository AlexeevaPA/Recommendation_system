from pydantic import BaseModel
from typing import List

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

class Response(BaseModel):
    recommendations: List[PostGet]
    exp_group: str

    class Config:
        orm_mode = True