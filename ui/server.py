"""A small local UI: pick a model, ask a question, see the answer and its uncertainty.

Run from the project root:

    .venv/bin/python ui/server.py            # http://127.0.0.1:8765

It listens on localhost only. One model is held at a time; loading another replaces it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import engine as E  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
app = FastAPI(title="Uncertainty explorer")
ENGINE = E.Engine()


class LoadReq(BaseModel):
    model: str


class RunReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_tier: int = Field(2, ge=1, le=4)
    bioasq_prompt: bool = True
    max_new_tokens: int | None = Field(None, ge=1, le=512)


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/models")
def models():
    return {"models": E.model_list(),
            "tiers": E.TIER_NEEDS,
            "estimators": {t: [E.PRETTY.get(n, n) for n in names]
                           for t, names in E.bioasq.TIERS.items()}}


@app.get("/api/status")
def status():
    return ENGINE.status()


@app.post("/api/load")
def load(req: LoadReq):
    try:
        ENGINE.start_load(req.model)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    return ENGINE.status()


@app.get("/api/random_question")
def random_question():
    q = E.QUESTION_BANK.random()
    if q is None:
        raise HTTPException(404, "no BioASQ run in outputs/ to draw from")
    return q


@app.post("/api/run")
def run(req: RunReq):
    # A plain def: FastAPI runs it in a worker thread, so a long generation does not
    # block /api/status. The engine's lock keeps generations one at a time.
    try:
        return ENGINE.run(req.prompt, req.max_tier, req.bioasq_prompt, req.max_new_tokens)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
