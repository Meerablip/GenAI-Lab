"""Interactive multimodal application (Gradio).

Tab 1  Understand an image : BLIP caption + optional BLIP-VQA answer.
Tab 2  Generate an image   : SD-Turbo text-to-image, then BLIP re-captions it and CLIP scores it against the prompt.

Run locally:  python app.py --port 7860
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multimodal import MultimodalToolkit  # noqa: E402

TOOLKIT = MultimodalToolkit()


def describe(image, question):
    if image is None:
        raise gr.Error("Upload an image first.")
    caption = TOOLKIT.caption(image)
    answer = TOOLKIT.vqa(image, question) if question and question.strip() else ""
    return caption, answer


def generate(prompt, steps, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")
    image = TOOLKIT.generate(prompt, int(steps), int(seed))
    caption = TOOLKIT.caption(image)
    score = float(TOOLKIT.clip_similarity([image], [prompt])[0, 0])
    return image, caption, round(score, 4)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Practical 4 - Multimodal AI") as demo:
        gr.Markdown("# Multimodal AI application\nVision-language understanding (BLIP, CLIP) "
                    "and text-to-image generation (SD-Turbo) in one interface. CPU inference: expect seconds per request.")
        with gr.Tab("Understand an image"):
            image = gr.Image(type="pil", label="Image")
            question = gr.Textbox(label="Question about the image (optional)")
            describe_btn = gr.Button("Describe")
            caption = gr.Textbox(label="BLIP caption")
            answer = gr.Textbox(label="BLIP-VQA answer")
            describe_btn.click(describe, [image, question], [caption, answer], api_name="describe")
        with gr.Tab("Generate an image"):
            prompt = gr.Textbox(label="Prompt", value="a lighthouse on a rocky cliff at sunset")
            steps = gr.Slider(1, 4, value=2, step=1, label="Sampling steps")
            seed = gr.Number(value=42, precision=0, label="Seed")
            generate_btn = gr.Button("Generate")
            out_image = gr.Image(label="SD-Turbo output")
            out_caption = gr.Textbox(label="BLIP caption of the generated image")
            out_clip = gr.Number(label="CLIP cosine(prompt, image)")
            generate_btn.click(generate, [prompt, steps, seed], [out_image, out_caption, out_clip], api_name="generate")
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_demo().launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
