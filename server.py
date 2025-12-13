#!/usr/bin/env python

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

import tempfile
import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# your existing search function
from cbf_to_vector import search_similar_cbf

app = FastAPI()

# initialize Qdrant client once
qdrant_client = QdrantClient(host="localhost", port=6333)

@app.post("/search_cbf/")
async def search_cbf_file(
    file: UploadFile = File(...),
    method: str = "pixel",
    size: int = 224,
    limit: int = 5,
    collection_name: str = "cbf_images"
):
    print("### search", file, method)
    # validate file type
    if not file.filename.lower().endswith(".cbf"):
        raise HTTPException(status_code=400, detail="Only CBF files allowed")

    # save uploaded file temporarily
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # call your search function
        hits = search_similar_cbf(
            cbf_path=tmp_path,
            client=qdrant_client,
            collection_name=collection_name,
            method=method,
            size=size,
            limit=limit,
        )

        # format response
        results = [
            {
                "id": h.id,
                "score": h.score,
                "path": h.payload.get("path")
            }
            for h in hits
        ]

        return {"results": results}

    finally:
        # cleanup file
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.get("/search_cbf_path/{file_path:path}")
def search_cbf_by_path(
    file_path: str,
    method: str = "pixel",
    size: int = 224,
    limit: int = 5,
    collection_name: str = "cbf_images",
):
    """
    Search for similar images from Qdrant given a local path to a CBF file.
    """
    # check file exists
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # call existing search function
    hits = search_similar_cbf(
        cbf_path=file_path,
        client=qdrant_client,
        collection_name=collection_name,
        method=method,
        size=size,
        limit=limit,
    )

    # format response
    results = [
        {
            "id": h.id,
            "score": h.score,
            "path": h.payload.get("path"),
        }
        for h in hits
    ]

    return {"results": results}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",  # or 127.0.0.1
        port=8111,
        log_level="info",
    )
