"""The website's backend. Run from the repo folder:  uvicorn app.main:app --port 8000"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.predictor import score_essay

app = FastAPI(title="Essay Scoring")


class EssayIn(BaseModel):
    essay: str


@app.post("/score")
def score(req: EssayIn):
    if len(req.essay.split()) < 10:
        raise HTTPException(status_code=400, detail="The essay must have at least 10 words.")
    return {"traits": score_essay(req.essay)}


# Serve the web page (index.html, style.css, app.js) at "/"
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
