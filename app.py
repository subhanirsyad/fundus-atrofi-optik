import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

import pipeline

BATAS_UKURAN = 25 * 1024 * 1024
HALAMAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

pipeline.muat_model()
app = FastAPI()


@app.get("/")
def halaman():
    return FileResponse(HALAMAN)


@app.get("/api/status")
def status():
    return {"siap": pipeline.GALAT_MUAT is None, "galat": pipeline.GALAT_MUAT}


@app.post("/api/analisis")
async def analisis(request: Request):
    if pipeline.GALAT_MUAT:
        raise HTTPException(503, "Bobot model belum ada di folder models/.")
    data = await request.body()
    if not data:
        raise HTTPException(400, "Tidak ada file yang dikirim.")
    if len(data) > BATAS_UKURAN:
        raise HTTPException(413, "File lebih dari 25 MB.")
    try:
        return await run_in_threadpool(pipeline.analisis, data)
    except ValueError as e:
        raise HTTPException(400, str(e))
