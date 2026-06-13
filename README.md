# Cross-Reference QA Dataset Builder

Builds a benchmark dataset for evaluating RAG systems on cross-reference questions
across NumPy and Pandas documentation.

## What this produces

| File | Description |
|---|---|
| `output/dataset.json` | Full dataset — 150-200 questions with ground truth |
| `output/dataset.csv` | Same data in flat CSV format |
| `output/verification_sample.json` | 50 randomly selected questions for manual review |

Each question requires following at least one explicit cross-reference to answer correctly.
Vector RAG systems that stop at the first retrieved page will fail these questions.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Run the pipeline

Run the four scripts in order:

```bash
python scrape.py    # Stage 1: crawl NumPy + Pandas docs (~10-20 min)
python parse.py     # Stage 2: extract cross-reference pairs (~1 min)
python generate.py  # Stage 3: generate questions via Gemini (~10-15 min)
python package.py   # Stage 4: package final dataset (~5 sec)
```

Each script prints a summary when it finishes so you can verify before moving on.

## Dataset schema

Each question in `dataset.json` has these fields:

```json
{
  "id": "q0001",
  "question": "How does np.einsum interact with broadcasting rules?",
  "ground_truth_answer": "According to the numpy.broadcast docs...",
  "answer_location": "target",
  "reasoning": "The einsum page mentions broadcasting but doesn't explain the rules",
  "library": "numpy",
  "source_url": "https://numpy.org/doc/stable/reference/generated/numpy.einsum.html",
  "source_title": "numpy.einsum",
  "target_url": "https://numpy.org/doc/stable/user/basics.broadcasting.html",
  "target_title": "Broadcasting",
  "difficulty": "single_hop",
  "flagged_for_review": false,
  "generated_by": "gemini-1.5-flash",
  "created_at": "2026-05-15T..."
}
```

## Difficulty levels

- `single_hop` — answer requires following ONE cross-reference (source → target)
- `multi_hop` — answer requires following TWO OR MORE cross-references

## Manual verification

Open `output/verification_sample.json` and for each question:

1. Read the question
2. Open `source_url` — confirm you **cannot** answer from this page alone
3. Open `target_url` — confirm the `ground_truth_answer` is accurate
4. Check the `difficulty` label is correct
5. Mark PASS / FAIL

Aim for >85% pass rate before using the dataset in evaluations.
