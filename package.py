"""
package.py — Stage 4: Package the final dataset

What this script does:
- Reads output/generated_questions.json produced by generate.py
- Adds a `flagged_for_review` field — 50 random questions are marked True
  so you know which ones to manually verify
- Writes three output files:
    * output/dataset.json         — full structured dataset (all questions)
    * output/dataset.csv          — flat CSV version for easy spreadsheet inspection
    * output/verification_sample.json — only the 50 flagged questions

The verification_sample.json is the file you manually review to validate
that questions genuinely require cross-reference following and ground truth
answers are accurate.

Run this last, after generate.py.
"""

import json
import csv
import random
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_FILE           = Path("output/generated_questions.json")
OUTPUT_DATASET_JSON  = Path("output/dataset.json")
OUTPUT_DATASET_CSV   = Path("output/dataset.csv")
OUTPUT_VERIFY_JSON   = Path("output/verification_sample.json")

# How many questions to flag for manual review
VERIFICATION_SAMPLE_SIZE = 50

# ── Helpers ────────────────────────────────────────────────────────────────────

def build_dataset_record(q: dict, flagged: bool) -> dict:
    """
    Transform a raw generated question into the final dataset record schema.
    This is the canonical shape of each record in dataset.json.
    """
    return {
        # Unique identifier
        "id": q["id"],

        # Core question data
        "question":             q["question"],
        "ground_truth_answer":  q["ground_truth_answer"],

        # Where the answer comes from
        "answer_location":      q.get("answer_location", "target"),
        "reasoning":            q.get("reasoning", ""),
        "difficulty_justification": q.get("difficulty_justification", ""),

        # Source document metadata
        "library":              q["library"],
        "source_url":           q["source_url"],
        "source_title":         q["source_title"],

        # Target document metadata (where the answer lives)
        "target_url":           q["target_url"],
        "target_title":         q["target_title"],

        # Difficulty classification
        "difficulty":           q["difficulty"],
        # single_hop: one cross-reference must be followed
        # multi_hop:  two or more cross-references must be followed

        # Whether this question is in the manual verification sample
        "flagged_for_review":   flagged,

        # Dataset provenance
        "generated_by":         "gemini-1.5-flash",
        "created_at":           datetime.utcnow().isoformat() + "Z",
    }


def write_csv(records: list[dict], filepath: Path):
    """Write dataset records to CSV. Flattens all fields to string columns."""
    if not records:
        return

    fieldnames = list(records[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            # Flatten any non-string values for CSV compatibility
            row = {k: str(v) for k, v in record.items()}
            writer.writerow(row)


def print_dataset_stats(records: list[dict]):
    """Print a breakdown of the dataset composition."""
    numpy_recs  = [r for r in records if r["library"] == "numpy"]
    pandas_recs = [r for r in records if r["library"] == "pandas"]
    single_recs = [r for r in records if r["difficulty"] == "single_hop"]
    multi_recs  = [r for r in records if r["difficulty"] == "multi_hop"]
    flagged     = [r for r in records if r["flagged_for_review"]]

    print(f"\n  Dataset composition:")
    print(f"    Total questions      : {len(records)}")
    print(f"    NumPy questions      : {len(numpy_recs)} ({100*len(numpy_recs)//len(records)}%)")
    print(f"    Pandas questions     : {len(pandas_recs)} ({100*len(pandas_recs)//len(records)}%)")
    print(f"    Single-hop           : {len(single_recs)} ({100*len(single_recs)//len(records)}%)")
    print(f"    Multi-hop            : {len(multi_recs)} ({100*len(multi_recs)//len(records)}%)")
    print(f"    Flagged for review   : {len(flagged)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)  # reproducibility — same 50 questions flagged every run

    # Load generated questions
    with open(INPUT_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} generated questions.")

    if len(questions) < VERIFICATION_SAMPLE_SIZE:
        print(
            f"WARNING: Only {len(questions)} questions available — "
            f"flagging all for review instead of {VERIFICATION_SAMPLE_SIZE}."
        )
        sample_size = len(questions)
    else:
        sample_size = VERIFICATION_SAMPLE_SIZE

    # Select which question IDs to flag for manual review
    # Stratified: pick proportionally from each library × difficulty bucket
    buckets = {
        ("numpy",  "single_hop"): [],
        ("numpy",  "multi_hop"):  [],
        ("pandas", "single_hop"): [],
        ("pandas", "multi_hop"):  [],
    }
    for q in questions:
        key = (q["library"], q["difficulty"])
        if key in buckets:
            buckets[key].append(q["id"])

    flagged_ids = set()
    # Distribute 50 slots proportionally across buckets
    total_q = len(questions)
    for key, ids in buckets.items():
        proportion = len(ids) / total_q
        n_from_bucket = max(1, round(sample_size * proportion))
        flagged_ids.update(random.sample(ids, min(n_from_bucket, len(ids))))

    # Top up or trim to exactly sample_size
    all_ids = [q["id"] for q in questions]
    remaining = [id_ for id_ in all_ids if id_ not in flagged_ids]
    while len(flagged_ids) < sample_size and remaining:
        flagged_ids.add(remaining.pop(0))
    flagged_ids = set(list(flagged_ids)[:sample_size])

    # Build final dataset records
    records = [
        build_dataset_record(q, flagged=(q["id"] in flagged_ids))
        for q in questions
    ]

    # ── Write dataset.json ───────────────────────────────────────────────────
    with open(OUTPUT_DATASET_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "description": (
                        "Cross-reference QA benchmark for NumPy and Pandas documentation. "
                        "Each question requires following at least one explicit cross-reference "
                        "to answer correctly — vector RAG systems that stop at the first page "
                        "will fail these questions."
                    ),
                    "total_questions":  len(records),
                    "libraries":        ["numpy", "pandas"],
                    "difficulty_types": ["single_hop", "multi_hop"],
                    "generated_by":     "gemini-1.5-flash",
                    "created_at":       datetime.utcnow().isoformat() + "Z",
                    "verification_sample_size": sample_size,
                },
                "questions": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ── Write dataset.csv ────────────────────────────────────────────────────
    write_csv(records, OUTPUT_DATASET_CSV)

    # ── Write verification_sample.json ───────────────────────────────────────
    verification_records = [r for r in records if r["flagged_for_review"]]

    with open(OUTPUT_VERIFY_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "description": (
                        f"Manual verification sample — {sample_size} questions randomly selected "
                        "from the full dataset for human review. For each question, verify: "
                        "(1) the question cannot be answered from the source page alone, "
                        "(2) the ground_truth_answer is accurate and drawn from the target page, "
                        "(3) the difficulty label (single_hop / multi_hop) is correct."
                    ),
                    "sample_size":  sample_size,
                    "review_instructions": {
                        "step_1": "Read the question.",
                        "step_2": "Open source_url — confirm the question cannot be answered from this page.",
                        "step_3": "Open target_url — confirm the ground_truth_answer is accurate.",
                        "step_4": "Check difficulty: single_hop = one link to follow, multi_hop = two or more.",
                        "step_5": "Mark PASS / FAIL and note any issues.",
                    },
                },
                "questions": verification_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ── Stage 4 Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STAGE 4 COMPLETE — Dataset Packaging Summary")
    print("=" * 55)
    print_dataset_stats(records)
    print(f"\n  Output files:")
    print(f"    {OUTPUT_DATASET_JSON}    — full dataset")
    print(f"    {OUTPUT_DATASET_CSV}     — flat CSV version")
    print(f"    {OUTPUT_VERIFY_JSON}  — {sample_size} questions for manual review")
    print("=" * 55)
    print("\n✓ Dataset build complete.")
    print(f"\nNext steps:")
    print(f"  1. Open {OUTPUT_VERIFY_JSON} and manually check ~50 questions")
    print(f"  2. Load {OUTPUT_DATASET_JSON} into your RAG evaluation harness")
    print(f"  3. Run PageIndex and FAISS against dataset.json")
    print(f"  4. Compare accuracy on cross-reference questions")


if __name__ == "__main__":
    main()
