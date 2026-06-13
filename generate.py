"""
generate.py — Stage 3: Generate questions using Gemini 2.5 Flash

What this script does:
- Reads output/cross_ref_pairs.json produced by parse.py
- For each cross-reference pair, calls Gemini to generate a question that:
    * Cannot be answered from the source page alone
    * Requires following the cross-reference to the target page
    * Has a ground truth answer pulled from the target page
- Generates 150-200 questions total, balanced across NumPy/Pandas and hop types
- Each question gets a unique ID, difficulty label, and ground truth
- Saves output/generated_questions.json for use by package.py

Requires: GEMINI_API_KEY in a .env file in this directory.
Run this after parse.py.
"""

import json
import os
import random
import time
import re
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# ── Configuration ──────────────────────────────────────────────────────────────

load_dotenv()

INPUT_FILE  = Path("output/cross_ref_pairs.json")
OUTPUT_FILE = Path("output/generated_questions.json")

TARGET_MIN          = 150
TARGET_MAX          = 200
TARGET_TOTAL        = 175
QUESTIONS_PER_PAIR  = 1
API_DELAY           = 1.5
NUMPY_FRACTION      = 0.5
SINGLE_HOP_FRACTION = 0.65

# ── Gemini Setup ───────────────────────────────────────────────────────────────

def setup_gemini() -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Create a .env file with: GEMINI_API_KEY=your_key_here"
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("✓ Gemini 2.5 Flash initialised")
    return model


# ── Prompt Templates ───────────────────────────────────────────────────────────

QUESTION_PROMPT = """You are building a benchmark dataset to test RAG (Retrieval Augmented Generation) systems.

Your task: generate ONE question that tests whether a retrieval system correctly follows a cross-reference between two documentation pages.

CRITICAL REQUIREMENT: The question must be answerable ONLY by reading the TARGET page — NOT from the source page alone. A system that reads only the source page and stops should get the answer wrong.

---
SOURCE PAGE: {source_title}
URL: {source_url}

SOURCE CONTENT (excerpt):
{source_text}

---
TARGET PAGE (reached by following the cross-reference): {target_title}
URL: {target_url}

TARGET CONTENT (excerpt — this is where the answer lives):
{target_text}

---
Difficulty: {difficulty}
- single_hop: question requires reading ONE cross-reference (source → target)
- multi_hop: question requires following TWO OR MORE cross-references

---
Respond with ONLY a JSON object in this exact format (no markdown, no backticks):
{{
  "question": "The question text here",
  "ground_truth_answer": "A 2-4 sentence answer drawn directly from the target page content",
  "answer_location": "target",
  "reasoning": "One sentence explaining why the source page alone cannot answer this",
  "difficulty_justification": "One sentence explaining why this is single_hop or multi_hop"
}}

The question should be the kind a developer would genuinely ask. Make it specific and technical.
"""


# ── Generation Logic ───────────────────────────────────────────────────────────

def generate_question(model: genai.GenerativeModel, pair: dict) -> dict | None:
    prompt = QUESTION_PROMPT.format(
        source_title = pair["source_title"],
        source_url   = pair["source_url"],
        source_text  = pair["source_text"],
        target_title = pair["target_title"],
        target_url   = pair["target_url"],
        target_text  = pair["target_text"],
        difficulty   = pair["difficulty"],
    )

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        # Strip markdown fences if the model added them anyway
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        data = json.loads(raw_text)

        required_fields = ["question", "ground_truth_answer", "answer_location", "reasoning", "difficulty_justification"]
        if not all(field in data for field in required_fields):
            print(f"  ✗ Missing fields in response, skipping")
            return None

        data["source_url"]   = pair["source_url"]
        data["source_title"] = pair["source_title"]
        data["target_url"]   = pair["target_url"]
        data["target_title"] = pair["target_title"]
        data["library"]      = pair["library"]
        data["difficulty"]   = pair["difficulty"]

        return data

    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parse error: {e}")
        print(f"  RAW (first 500 chars): {raw_text[:500]}")
        return None
    except Exception as e:
        print(f"  ✗ API error: {e}")
        return None


def sample_pairs(pairs: list[dict], target_count: int) -> list[dict]:
    numpy_pairs  = [p for p in pairs if p["library"] == "numpy"]
    pandas_pairs = [p for p in pairs if p["library"] == "pandas"]

    n_numpy  = int(target_count * NUMPY_FRACTION)
    n_pandas = target_count - n_numpy

    def split_by_hop(pool: list[dict], n: int) -> list[dict]:
        single = [p for p in pool if p["difficulty"] == "single_hop"]
        multi  = [p for p in pool if p["difficulty"] == "multi_hop"]
        n_single = int(n * SINGLE_HOP_FRACTION)
        n_multi  = n - n_single
        sampled  = random.sample(single, min(n_single, len(single)))
        sampled += random.sample(multi,  min(n_multi,  len(multi)))
        return sampled

    sample = split_by_hop(numpy_pairs, n_numpy) + split_by_hop(pandas_pairs, n_pandas)
    random.shuffle(sample)
    return sample[:target_count]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)

    with open(INPUT_FILE, encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} cross-reference pairs.")

    if len(pairs) < 10:
        print("ERROR: Too few pairs found. Check that parse.py ran correctly.")
        return

    model = setup_gemini()

    sampled_pairs = sample_pairs(pairs, TARGET_TOTAL)
    print(f"Sampled {len(sampled_pairs)} pairs to generate questions from.")

    # ── DEBUG: remove this line after confirming output is correct ──

    questions = []
    failed    = 0

    for i, pair in enumerate(sampled_pairs):
        print(f"  [{i+1}/{len(sampled_pairs)}] Generating: {pair['source_title']} → {pair['target_title']}")

        question = generate_question(model, pair)

        if question is not None:
            question["id"] = f"q{len(questions)+1:04d}"
            questions.append(question)
        else:
            failed += 1

        if len(questions) >= TARGET_MAX:
            print(f"  Reached target of {TARGET_MAX} questions — stopping early.")
            break

        time.sleep(API_DELAY)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    numpy_qs  = [q for q in questions if q["library"] == "numpy"]
    pandas_qs = [q for q in questions if q["library"] == "pandas"]
    single_qs = [q for q in questions if q["difficulty"] == "single_hop"]
    multi_qs  = [q for q in questions if q["difficulty"] == "multi_hop"]

    print("\n" + "=" * 55)
    print("STAGE 3 COMPLETE — Question Generation Summary")
    print("=" * 55)
    print(f"  Questions generated  : {len(questions)}")
    print(f"  Generation failures  : {failed}")
    print(f"  NumPy questions      : {len(numpy_qs)}")
    print(f"  Pandas questions     : {len(pandas_qs)}")
    print(f"  Single-hop           : {len(single_qs)}")
    print(f"  Multi-hop            : {len(multi_qs)}")
    print(f"  Output saved to      : {OUTPUT_FILE}")
    print("=" * 55)
    print("Next step: run  python package.py")


if __name__ == "__main__":
    main()