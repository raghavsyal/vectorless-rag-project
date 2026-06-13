"""
pipeline_faiss.py — Pipeline A: Vector RAG using FAISS

What this script does:
- Loads scraped_pages.json and chunks each page into ~300 word segments
- Embeds chunks using sentence-transformers/all-MiniLM-L6-v2
- Builds a FAISS index over all chunks
- For each question in dataset.json, retrieves top-k chunks and sends to Gemini for answering
- Records answer, latency, token usage, and cost per query
- Saves results to output/results_faiss.json
"""

import json
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

PAGES_FILE   = Path("output/scraped_pages.json")
DATASET_FILE = Path("output/dataset.json")
OUTPUT_FILE  = Path("output/results_faiss.json")
INDEX_FILE   = Path("output/faiss.index")
CHUNKS_FILE  = Path("output/faiss_chunks.json")

CHUNK_SIZE   = 300   # words per chunk
CHUNK_OVERLAP = 50   # word overlap between chunks
TOP_K        = 5     # chunks to retrieve per query

EMBED_MODEL  = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

# Cost per 1M tokens (Gemini 2.5 Flash as of 2025)
COST_INPUT_PER_1M  = 0.075
COST_OUTPUT_PER_1M = 0.30

# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_page(page: dict) -> list[dict]:
    """Split a page's body text into overlapping word-window chunks."""
    words = page["body_text"].split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "text":    chunk_text,
            "url":     page["url"],
            "title":   page["title"],
            "library": page["library"],
            "chunk_id": f"{page['url']}::{start}",
        })
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_chunks(pages: list[dict]) -> list[dict]:
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_page(page))
    return all_chunks


# ── Index ──────────────────────────────────────────────────────────────────────

def build_index(chunks: list[dict], model: SentenceTransformer) -> faiss.IndexFlatIP:
    """Embed all chunks and build a FAISS inner-product index."""
    print(f"  Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  ✓ FAISS index built — {index.ntotal} vectors, dim={dim}")
    return index, embeddings


def retrieve(question: str, index: faiss.IndexFlatIP, chunks: list[dict],
             model: SentenceTransformer, k: int = TOP_K) -> list[dict]:
    """Embed the question and return top-k matching chunks."""
    q_emb = model.encode([question], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


# ── Answer Generation ──────────────────────────────────────────────────────────

ANSWER_PROMPT = """You are a technical documentation assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer, say "I cannot answer from the provided context."

Context:
{context}

Question: {question}

Answer concisely in 2-4 sentences."""


def answer_question(question: str, chunks: list[dict],
                    gemini: genai.GenerativeModel) -> dict:
    """Send retrieved chunks to Gemini and return answer + usage stats."""
    context = "\n\n---\n\n".join(
        f"[{c['title']}]\n{c['text']}" for c in chunks
    )
    prompt = ANSWER_PROMPT.format(context=context, question=question)

    t0 = time.time()
    response = gemini.generate_content(prompt)
    latency = time.time() - t0

    answer = response.text.strip()

    # Extract token counts from usage metadata
    usage = response.usage_metadata
    input_tokens  = usage.prompt_token_count     if usage else 0
    output_tokens = usage.candidates_token_count if usage else 0

    cost = (input_tokens / 1_000_000 * COST_INPUT_PER_1M +
            output_tokens / 1_000_000 * COST_OUTPUT_PER_1M)

    return {
        "answer":        answer,
        "latency_s":     round(latency, 3),
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "cost_usd":      round(cost, 6),
        "retrieved_chunks": [{"url": c["url"], "title": c["title"], "score": c["score"]} for c in chunks],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load data
    with open(PAGES_FILE, encoding="utf-8") as f:
        pages = json.load(f)
    with open(DATASET_FILE, encoding="utf-8") as f:
        dataset = json.load(f)
        questions = dataset["questions"] # Target the array inside the JSON

    print(f"Loaded {len(pages)} pages, {len(questions)} questions.")

    # Build or load chunks + index
    embed_model = SentenceTransformer(EMBED_MODEL)

    if INDEX_FILE.exists() and CHUNKS_FILE.exists():
        print("Loading existing FAISS index and chunks...")
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            chunks = json.load(f)
        index = faiss.read_index(str(INDEX_FILE))
        print(f"  ✓ Loaded {len(chunks)} chunks, {index.ntotal} vectors")
    else:
        print("Building chunks and FAISS index...")
        chunks = build_chunks(pages)
        print(f"  ✓ {len(chunks)} chunks created from {len(pages)} pages")
        index, _ = build_index(chunks, embed_model)
        # Save for reuse
        faiss.write_index(index, str(INDEX_FILE))
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        print("  ✓ Index and chunks saved to disk")

    # Set up Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini = genai.GenerativeModel(GEMINI_MODEL)

    # Run evaluation
    results = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['question'][:80]}...")

        retrieved = retrieve(q["question"], index, chunks, embed_model)
        result    = answer_question(q["question"], retrieved, gemini)

        results.append({
            "id":                q["id"],
            "question":          q["question"],
            "ground_truth":      q["ground_truth_answer"],
            "library":           q["library"],
            "difficulty":        q["difficulty"],
            "pipeline":          "faiss",
            **result,
        })

        time.sleep(2.0)   # stay under RPM limit

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    total_cost    = sum(r["cost_usd"] for r in results)
    avg_latency   = sum(r["latency_s"] for r in results) / len(results)
    avg_input_tok = sum(r["input_tokens"] for r in results) / len(results)

    print("\n" + "=" * 55)
    print("FAISS PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Questions answered : {len(results)}")
    print(f"  Avg latency        : {avg_latency:.2f}s")
    print(f"  Avg input tokens   : {avg_input_tok:.0f}")
    print(f"  Total cost (USD)   : ${total_cost:.4f}")
    print(f"  Output saved to    : {OUTPUT_FILE}")
    print("=" * 55)
    print("Next step: run  python pipeline_pageindex.py")


if __name__ == "__main__":
    main()