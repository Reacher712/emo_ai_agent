"""Original Z-Image-Turbo inference, without quantization or LoRA."""
import argparse
import gc
import json
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
# Keep Hugging Face caches on the project disk, not the user's system drive.
os.environ.setdefault("HF_HOME", str(ROOT / "models" / ".hf-cache"))

import torch
from diffusers import ZImagePipeline


class ZImageEngine:
    def __init__(self, model_dir=None, offload="staged"):
        self.model_dir = Path(model_dir or ROOT / "models" / "Z-Image-Turbo")
        self.offload = offload
        self.pipe = None
        self.lock = Lock()

    def load(self):
        if self.pipe is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install the CUDA PyTorch build first")
        if not (self.model_dir / "model_index.json").exists():
            raise FileNotFoundError("Official weights not downloaded; run download_model.py")
        print(f"Loading original weights on {torch.cuda.get_device_name(0)} ({self.offload} offload)", flush=True)
        components = {"text_encoder": None, "tokenizer": None} if self.offload == "staged" else {}
        pipe = ZImagePipeline.from_pretrained(
            str(self.model_dir), dtype=torch.bfloat16,
            low_cpu_mem_usage=True, local_files_only=True, **components,
        )
        # Text encoder, transformer and VAE take turns on the GPU. Sequential
        # offload is available when desktop usage leaves insufficient VRAM.
        if self.offload in {"model", "staged"}:
            pipe.enable_model_cpu_offload()
        elif self.offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.vae.enable_tiling()
        self.pipe = pipe

    def _encode_staged(self, prompt):
        # On a 32 GB host the full text+image pipeline can exceed available RAM.
        # Release the previous image pipeline before loading the text encoder.
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        encoder = ZImagePipeline.from_pretrained(
            str(self.model_dir), transformer=None, vae=None,
            dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True,
        )
        encoder.to("cuda")
        with torch.inference_mode():
            embeds, _ = encoder.encode_prompt(prompt, device=torch.device("cuda"),
                                              do_classifier_free_guidance=False)
        # Only tiny prompt embeddings survive; the original text model is freed.
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        return embeds

    def generate(self, prompt, *, width=768, height=768, seed=42, steps=9):
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if not 256 <= width <= 1536 or not 256 <= height <= 1536 or width % 16 or height % 16:
            raise ValueError("Dimensions must be multiples of 16 between 256 and 1536")
        if not 1 <= steps <= 20 or not 0 <= seed < 2**63:
            raise ValueError("Invalid steps or seed")
        with self.lock:
            start = perf_counter()
            embeddings = self._encode_staged(prompt) if self.offload == "staged" else None
            self.load()
            load_seconds = perf_counter() - start
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            try:
                with torch.inference_mode():
                    image = self.pipe(
                        prompt=None if embeddings is not None else prompt,
                        prompt_embeds=embeddings, width=width, height=height,
                        num_inference_steps=steps, guidance_scale=0.0,
                        generator=torch.Generator("cpu").manual_seed(seed),
                    ).images[0]
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                raise
            torch.cuda.synchronize()
            inference_seconds = perf_counter() - start
            output = ROOT / "outputs"
            output.mkdir(parents=True, exist_ok=True)
            artifact_id = uuid4().hex
            path = output / f"{artifact_id}.png"
            image.save(path)
            manifest = self.model_dir / "deployment_manifest.json"
            metadata = {
                "artifact_id": artifact_id, "image_path": str(path),
                "backend": "z-image-turbo", "prompt": prompt,
                "width": width, "height": height, "seed": seed,
                "steps": steps, "guidance_scale": 0.0, "dtype": "bfloat16",
                "offload": self.offload, "lora": None, "quantization": None,
                "load_seconds": round(load_seconds, 3),
                "inference_seconds": round(inference_seconds, 3),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                "model": json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None,
            }
            path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata


def main():
    parser = argparse.ArgumentParser(description="Original Z-Image-Turbo local generation")
    parser.add_argument("--prompt", default="Chinese ink wash landscape painting, distant misty mountains, a small bamboo grove, delicate brushwork, pale ink on rice paper, generous negative space, peaceful atmosphere, no text")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload", choices=["staged", "model", "sequential", "none"], default="staged")
    args = parser.parse_args()
    result = ZImageEngine(offload=args.offload).generate(
        args.prompt, width=args.width, height=args.height, steps=args.steps, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

