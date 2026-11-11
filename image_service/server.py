"""Local-only HTTP inference service; raw outputs have not passed agent review."""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from generate import ZImageEngine

app = FastAPI(title="Local original Z-Image-Turbo", version="0.1.0")
engine = ZImageEngine(offload=os.environ.get("ZIMAGE_OFFLOAD", "staged"))


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=6000)
    width: int = Field(default=768, ge=256, le=1536, multiple_of=16)
    height: int = Field(default=768, ge=256, le=1536, multiple_of=16)
    seed: int = Field(default=42, ge=0, lt=2**63)
    steps: int = Field(default=9, ge=1, le=20)


@app.get("/health")
def health():
    return {"service": "z-image-turbo", "loaded": engine.pipe is not None,
            "weights_present": (engine.model_dir / "deployment_manifest.json").exists(),
            "mock": False, "reviewed": False}


@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        return engine.generate(**request.model_dump())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

