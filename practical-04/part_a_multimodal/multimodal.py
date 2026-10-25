"""Multimodal toolkit: vision-language understanding (BLIP, CLIP) plus text-to-image generation (SD-Turbo).

All models run on CPU. Components are loaded lazily and each load is logged
(time, parameter count, process memory) so the experiment can report it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import psutil
import torch
from PIL import Image

CAPTION_ID = "Salesforce/blip-image-captioning-base"
VQA_ID = "Salesforce/blip-vqa-base"
CLIP_ID = "openai/clip-vit-base-patch32"
T2I_ID = "stabilityai/sd-turbo"
SBERT_ID = "sentence-transformers/all-MiniLM-L6-v2"


def n_params(*modules) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


@dataclass
class LoadRecord:
    key: str
    model_id: str
    role: str
    params_m: float
    load_s: float
    rss_gb_after: float


@dataclass
class MultimodalToolkit:
    """Lazy holder for every model used in Practical 4A."""

    t2i_size: int = 512
    load_log: list[LoadRecord] = field(default_factory=list)
    _cache: dict = field(default_factory=dict)

    # ---- loading -------------------------------------------------------------------
    def _load(self, key: str, model_id: str, role: str, builder):
        if key in self._cache:
            return self._cache[key]
        t0 = time.perf_counter()
        obj, modules = builder()
        elapsed = time.perf_counter() - t0
        self.load_log.append(LoadRecord(
            key, model_id, role, round(n_params(*modules) / 1e6, 1), round(elapsed, 2),
            round(psutil.Process().memory_info().rss / 1024**3, 3)))
        self._cache[key] = obj
        return obj

    @property
    def captioner(self):
        def build():
            from transformers import BlipForConditionalGeneration, BlipProcessor

            proc = BlipProcessor.from_pretrained(CAPTION_ID)
            model = BlipForConditionalGeneration.from_pretrained(CAPTION_ID).eval()
            return (proc, model), [model]
        return self._load("blip_caption", CAPTION_ID, "image captioning (VLM)", build)

    @property
    def vqa_model(self):
        def build():
            from transformers import BlipForQuestionAnswering, BlipProcessor

            proc = BlipProcessor.from_pretrained(VQA_ID)
            model = BlipForQuestionAnswering.from_pretrained(VQA_ID).eval()
            return (proc, model), [model]
        return self._load("blip_vqa", VQA_ID, "visual question answering (VLM)", build)

    @property
    def clip(self):
        def build():
            from transformers import CLIPModel, CLIPProcessor

            proc = CLIPProcessor.from_pretrained(CLIP_ID)
            model = CLIPModel.from_pretrained(CLIP_ID).eval()
            return (proc, model), [model]
        return self._load("clip", CLIP_ID, "image-text alignment (CLIP)", build)

    @property
    def text_embedder(self):
        def build():
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(SBERT_ID, device="cpu")
            return model, [model]
        return self._load("sbert", SBERT_ID, "sentence embeddings", build)

    @property
    def t2i(self):
        def build():
            from diffusers import AutoPipelineForText2Image

            try:  # the fp16 checkpoint is half the download; weights are up-cast to fp32 for CPU
                pipe = AutoPipelineForText2Image.from_pretrained(
                    T2I_ID, torch_dtype=torch.float32, variant="fp16")
            except Exception:
                pipe = AutoPipelineForText2Image.from_pretrained(T2I_ID, torch_dtype=torch.float32)
            pipe.set_progress_bar_config(disable=True)
            return pipe, [pipe.unet, pipe.text_encoder, pipe.vae]
        return self._load("sd_turbo", T2I_ID, "text-to-image diffusion", build)

    # ---- vision-language understanding ----------------------------------------------
    @torch.inference_mode()
    def caption(self, image: Image.Image, prefix: str | None = None) -> str:
        proc, model = self.captioner
        inputs = proc(images=image, text=prefix, return_tensors="pt") if prefix else proc(images=image, return_tensors="pt")
        ids = model.generate(**inputs, max_new_tokens=30, num_beams=5)
        return proc.decode(ids[0], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def vqa(self, image: Image.Image, question: str) -> str:
        proc, model = self.vqa_model
        inputs = proc(images=image, text=question, return_tensors="pt")
        ids = model.generate(**inputs, max_new_tokens=10)
        return proc.decode(ids[0], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def clip_similarity(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        """Cosine similarity matrix (len(images) x len(texts)) in CLIP's joint embedding space."""
        proc, model = self.clip
        inputs = proc(text=texts, images=images, return_tensors="pt", padding=True)
        out = model(**inputs)
        return (out.logits_per_image / model.logit_scale.exp()).cpu()

    @torch.inference_mode()
    def zero_shot(self, image: Image.Image, labels: list[str]) -> list[float]:
        """CLIP zero-shot classification: softmax over the scaled image-text similarities."""
        _, model = self.clip
        cos = self.clip_similarity([image], labels)[0]
        return torch.softmax(cos * model.logit_scale.exp().cpu(), dim=-1).tolist()

    @torch.inference_mode()
    def text_similarity(self, a: str, b: str) -> float:
        from sentence_transformers import util

        emb = self.text_embedder.encode([a, b], normalize_embeddings=True, convert_to_tensor=True)
        return float(util.cos_sim(emb[0], emb[1]))

    # ---- text-to-image -------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, prompt: str, steps: int = 4, seed: int = 42) -> Image.Image:
        """SD-Turbo is distilled for 1-4 steps and is sampled without classifier-free guidance."""
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        return self.t2i(prompt, num_inference_steps=int(steps), guidance_scale=0.0, generator=gen,
                        height=self.t2i_size, width=self.t2i_size).images[0]

    def load_report(self) -> list[dict]:
        return [r.__dict__ for r in self.load_log]
