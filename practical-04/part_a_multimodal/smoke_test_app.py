"""Launch the Gradio app headless and drive it through its HTTP API, as a user's browser would."""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import gradio  # noqa: E402
from gradio_client import Client, handle_file  # noqa: E402

from app import TOOLKIT, build_demo  # noqa: E402
from common import ResourceMonitor, load_json, write_json  # noqa: E402

PORT = 7860


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/part_a")
    args = ap.parse_args()
    out = Path(args.out)
    cfg = load_json(HERE / "tasks.json")["app_smoke"]

    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=PORT, prevent_thread_lock=True, quiet=True)
    client = Client(f"http://127.0.0.1:{PORT}", verbose=False)
    result: dict = {"gradio": gradio.__version__, "endpoints": ["/describe", "/generate"]}

    t0 = time.perf_counter()
    caption, answer = client.predict(handle_file(str(HERE / cfg["describe_image"])), cfg["describe_question"],
                                     api_name="/describe")
    result["describe"] = {"image": cfg["describe_image"], "question": cfg["describe_question"],
                          "caption": caption, "answer": answer, "seconds": round(time.perf_counter() - t0, 2)}
    print("describe ->", result["describe"], flush=True)

    with ResourceMonitor() as mon:
        image_path, caption, clip = client.predict(cfg["generate_prompt"], cfg["generate_steps"],
                                                   cfg["generate_seed"], api_name="/generate")
    dest = out / "app_smoke_generated.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(image_path, dest)
    result["generate"] = {"prompt": cfg["generate_prompt"], "steps": cfg["generate_steps"],
                          "seed": cfg["generate_seed"], "caption": caption, "clip_cosine": clip,
                          "seconds": mon.result["wall_s"], "peak_rss_gb": mon.result["peak_rss_gb"],
                          "image": dest.name}
    print("generate ->", result["generate"], flush=True)

    result["models_loaded"] = TOOLKIT.load_report()
    write_json(out / "app_smoke.json", result)
    demo.close()


if __name__ == "__main__":
    main()
