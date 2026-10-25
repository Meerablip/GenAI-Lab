"""Practical 4A: multimodal AI application = vision-language models + text-to-image generation.

Stages
  1. Load the models and record cost (time, parameters, memory).
  2. Vision-language understanding of five real photographs: captioning, VQA, CLIP zero-shot.
  3. Text-to-image generation: 6 prompts x 3 sampling-step settings with SD-Turbo.
  4. Close the loop: the VLMs inspect the generated images and are scored against the original prompts.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # common.py
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from common import ResourceMonitor, contains_key, load_json, snapshot_env, write_json  # noqa: E402
from multimodal import CAPTION_ID, CLIP_ID, SBERT_ID, T2I_ID, VQA_ID, MultimodalToolkit  # noqa: E402

PACKAGES = ["torch", "torchvision", "transformers", "diffusers", "accelerate", "sentence-transformers",
            "huggingface-hub", "safetensors", "pillow", "numpy", "gradio", "gradio-client", "psutil"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, round(time.perf_counter() - t0, 3)


def ask_checks(tk: MultimodalToolkit, image: Image.Image, checks: list[dict]) -> list[dict]:
    rows = []
    for c in checks:
        pred, secs = timed(tk.vqa, image, c["q"])
        rows.append({"question": c["q"], "gold": c["answers"], "predicted": pred, "seconds": secs,
                     "correct": any(contains_key(pred, g) for g in c["answers"])})
    return rows


# --------------------------------------------------------------------------- stage 2
def run_real_images(tk: MultimodalToolkit, tasks: dict, smoke: bool) -> list[dict]:
    labels = tasks["clip_labels"]
    items = tasks["real_images"][:1] if smoke else tasks["real_images"]
    rows = []
    for item in items:
        log(f"real image: {item['id']}")
        image = Image.open(HERE / item["file"]).convert("RGB")
        caption, cap_s = timed(tk.caption, image)
        prefixed, _ = timed(tk.caption, image, "a photography of")
        cos = float(tk.clip_similarity([image], [caption])[0, 0])
        probs = tk.zero_shot(image, labels)
        top = int(np.argmax(probs))
        qa = ask_checks(tk, image, item["questions"])
        rows.append({
            "id": item["id"], "size": list(image.size), "caption": caption, "caption_seconds": cap_s,
            "caption_with_prefix": prefixed, "clip_caption_cosine": round(cos, 4),
            "zero_shot_top_label": labels[top], "zero_shot_top_prob": round(probs[top], 4),
            "zero_shot_correct": top == item["truth_label"],
            "zero_shot_probs": {l: round(p, 4) for l, p in zip(labels, probs)},
            "vqa": qa,
        })
        log(f"  caption: {caption!r} | zero-shot: {labels[top]!r} ({probs[top]:.2f}) | "
            f"vqa {sum(r['correct'] for r in qa)}/{len(qa)}")
    return rows


# --------------------------------------------------------------------------- stage 3 + 4
def run_generation(tk: MultimodalToolkit, tasks: dict, out: Path, smoke: bool) -> dict:
    prompts = tasks["generation_prompts"][:1] if smoke else tasks["generation_prompts"]
    steps_list = [1] if smoke else tasks["steps"]
    seed0 = tasks["base_seed"]
    img_dir = out / "generated"
    img_dir.mkdir(parents=True, exist_ok=True)

    log("warming up the diffusion pipeline (1 step, discarded)")
    _, warm_s = timed(tk.generate, "a photo of a tree", 1, 0)

    images: dict[tuple[str, int], Image.Image] = {}
    records = []
    for pi, p in enumerate(prompts):
        for steps in steps_list:
            seed = seed0 + pi
            with ResourceMonitor() as mon:
                image = tk.generate(p["prompt"], steps, seed)
            image.save(img_dir / f"{p['id']}_s{steps}.png")
            images[(p["id"], steps)] = image
            records.append({"prompt_id": p["id"], "prompt": p["prompt"], "steps": steps, "seed": seed, **mon.result})
            log(f"generated {p['id']} @ {steps} step(s): {mon.result['wall_s']:.1f}s, "
                f"peak RSS {mon.result['peak_rss_gb']:.2f} GB, {mon.result['cpu_cores_used']:.1f} cores")

    # ---- stage 4: inspect every generated image with the vision-language models
    log("round trip: captioning, VQA and CLIP scoring of generated images")
    for rec in records:
        image = images[(rec["prompt_id"], rec["steps"])]
        prompt_task = next(p for p in prompts if p["id"] == rec["prompt_id"])
        caption, cap_s = timed(tk.caption, image)
        rec["caption"] = caption
        rec["caption_seconds"] = cap_s
        rec["caption_prompt_similarity"] = round(tk.text_similarity(rec["prompt"], caption), 4)
        rec["clip_prompt_cosine"] = round(float(tk.clip_similarity([image], [rec["prompt"]])[0, 0]), 4)
        rec["vqa"] = ask_checks(tk, image, prompt_task["checks"])

    # per-step aggregates + CLIP retrieval (does the matching prompt rank first among all prompts?)
    per_step = []
    for steps in steps_list:
        recs = [r for r in records if r["steps"] == steps]
        imgs = [images[(r["prompt_id"], steps)] for r in recs]
        sim = tk.clip_similarity(imgs, [r["prompt"] for r in recs]).numpy()
        vqa = [q for r in recs for q in r["vqa"]]
        per_step.append({
            "steps": steps,
            "mean_latency_s": round(float(np.mean([r["wall_s"] for r in recs])), 2),
            "mean_peak_rss_gb": round(float(np.mean([r["peak_rss_gb"] for r in recs])), 2),
            "mean_clip_cosine": round(float(np.mean([r["clip_prompt_cosine"] for r in recs])), 4),
            "mean_caption_similarity": round(float(np.mean([r["caption_prompt_similarity"] for r in recs])), 4),
            "vqa_correct": int(sum(q["correct"] for q in vqa)), "vqa_total": len(vqa),
            "clip_image_to_prompt_top1": round(float(np.mean(sim.argmax(axis=1) == np.arange(len(recs)))), 4),
            "clip_prompt_to_image_top1": round(float(np.mean(sim.argmax(axis=0) == np.arange(len(recs)))), 4),
            "clip_similarity_matrix": np.round(sim, 4).tolist(),  # rows: images, columns: prompts
        })
    return {"warmup_s": warm_s, "records": records, "per_step": per_step, "prompt_ids": [p["id"] for p in prompts]}


# --------------------------------------------------------------------------- output
def markdown_summary(res: dict) -> str:
    lines = ["## Practical 4A - multimodal pipeline", "", "### Models", "",
             "| model | role | params (M) | load (s) | RSS after (GB) |", "|---|---|---|---|---|"]
    for m in res["models"]:
        lines.append(f"| `{m['model_id']}` | {m['role']} | {m['params_m']} | {m['load_s']} | {m['rss_gb_after']} |")
    rows = res["real_images"]
    lines += ["", "### Real images", "", "| image | caption | zero-shot | VQA |", "|---|---|---|---|"]
    for r in rows:
        ok = sum(q["correct"] for q in r["vqa"])
        lines.append(f"| {r['id']} | {r['caption']} | {'ok' if r['zero_shot_correct'] else 'wrong'} | {ok}/{len(r['vqa'])} |")
    lines += ["", "### Text-to-image (SD-Turbo, CPU)", "",
              "| steps | latency (s) | CLIP cos | caption sim | VQA | CLIP top-1 |", "|---|---|---|---|---|---|"]
    for s in res["generation"]["per_step"]:
        lines.append(f"| {s['steps']} | {s['mean_latency_s']} | {s['mean_clip_cosine']} | {s['mean_caption_similarity']} "
                     f"| {s['vqa_correct']}/{s['vqa_total']} | {s['clip_image_to_prompt_top1']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/part_a")
    ap.add_argument("--smoke", action="store_true", help="1 image, 1 prompt, 1 step (wiring check)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_json(HERE / "tasks.json")

    result: dict = {"env": snapshot_env(PACKAGES), "smoke": args.smoke,
                    "model_ids": {"caption": CAPTION_ID, "vqa": VQA_ID, "clip": CLIP_ID, "t2i": T2I_ID, "sbert": SBERT_ID}}
    tk = MultimodalToolkit()

    log("loading vision-language models")
    with ResourceMonitor() as mon:
        _ = (tk.captioner, tk.vqa_model, tk.clip, tk.text_embedder)
    result["vlm_load_resources"] = mon.result

    log("warming up the VLMs (discarded)")
    first = Image.open(HERE / tasks["real_images"][0]["file"]).convert("RGB")
    _, warm_cap = timed(tk.caption, first)
    _, warm_vqa = timed(tk.vqa, first, "what is this?")
    result["vlm_warmup_s"] = {"caption": warm_cap, "vqa": warm_vqa}

    with ResourceMonitor() as mon:
        result["real_images"] = run_real_images(tk, tasks, args.smoke)
    result["real_images_resources"] = mon.result

    log("loading the text-to-image pipeline")
    with ResourceMonitor() as mon:
        _ = tk.t2i
    result["t2i_load_resources"] = mon.result

    with ResourceMonitor() as mon:
        result["generation"] = run_generation(tk, tasks, out, args.smoke)
    result["generation_resources"] = mon.result
    result["models"] = tk.load_report()
    result["torch_threads"] = torch.get_num_threads()

    write_json(out / "results.json", result)
    summary = markdown_summary(result)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    log(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
