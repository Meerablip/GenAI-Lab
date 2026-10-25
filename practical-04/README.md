# Practical 4

**(a)** Multimodal AI application using vision-language models and text-to-image generation.
**(b)** Retrieval-Augmented Generation (RAG) with embeddings, vector databases and semantic search.

Both parts run on CPU-only GitHub-hosted runners (`ubuntu-latest`) as two parallel jobs of
[`practical-04.yml`](../.github/workflows/practical-04.yml).

## Part A - `part_a_multimodal/`

| Component | Model |
|---|---|
| Captioning, VQA (VLM) | `Salesforce/blip-image-captioning-base`, `Salesforce/blip-vqa-base` |
| Image-text alignment | `openai/clip-vit-base-patch32` |
| Text-to-image | `stabilityai/sd-turbo` (diffusers, fp32 on CPU) |
| Sentence similarity | `sentence-transformers/all-MiniLM-L6-v2` |

* `multimodal.py` - the toolkit (lazy model loading, captioning, VQA, CLIP, generation).
* `run_experiment.py` - real-image understanding, a 6-prompt x 3-step generation grid, and a round-trip evaluation.
* `app.py` / `smoke_test_app.py` - the interactive Gradio app and a headless test that drives it through its HTTP API.

## Part B - `part_b_rag/`

* `knowledge_base/` - 25 short documents about a **fictional** institute, so the LLM cannot know the facts from training.
* `questions.json` - 24 questions (8 keyword, 14 paraphrased, 2 unanswerable) with gold documents and answer keys.
* `rag.py` / `run_experiment.py` - chunking, MiniLM and BGE embeddings, ChromaDB + FAISS + BM25 indexes,
  retrieval metrics, and answer generation with `llama3.2:3b` through Ollama
  (closed-book vs. RAG vs. oracle context).

## Run locally

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r part_a_multimodal/requirements.txt   # or part_b_rag/requirements.txt
python part_a_multimodal/run_experiment.py --out results/part_a --smoke
python part_b_rag/run_experiment.py --out results/part_b --skip-llm
```
