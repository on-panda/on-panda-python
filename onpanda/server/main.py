from typing import Union

from fastapi import FastAPI, APIRouter

FILE_SUFFIX = ".panda.json"

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "onPanda"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


router = APIRouter(prefix="/panda-api/v1")


@router.get("/")
def read_root():
    FILE_SUFFIX
    return {"Hello": "onPanda"}


app.include_router(router)
