"""
evaluate.py — Evaluation suite: PageIndex vs FAISS

What this script does:
- Loads results_faiss.json and results_pageindex.json
- Evaluates only questions answered by BOTH pipelines (fair comparison)
- Uses Cerebras gpt-oss-120b as LLM-as-judge to score each answer vs ground truth
- Computes accuracy, latency, token usage, and cost per query
- Breaks results down by difficulty (single_hop / multi_hop) and library
- For PageIndex: also computes navigation success rate (did it visit the target page?)
- Saves full results to output/evaluation_results.json
- Saves a clean summary to output/evaluation_summary.json
- Prints a formatted comparison table

Requires: CEREBRAS_API_KEY in .env
Run this after both pipelines have completed.
"""

import json
import os
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

FAISS_FILE     = Path("output/results_faiss.json")
PAGEINDEX_FILE = Path("output/results_pageindex.json")
DATASET_FILE   = Path("output/dataset.json")
OUTPUT_FULL    = Path("output/evaluation_results.json")
OUTPUT_SUMMARY = Path("output/evaluation_summary.json")

CEREBRAS_MODEL = "gpt-oss-120b"
API_DELAY      = 13.0

# ── Cerebras Judge Setup ───────────────────────────────────────────────────────

def setup_cerebras() -> Cerebras:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY not found in .env")
    return Cerebras(api_key=api_key)


# ── LLM-as-Judge ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating whether a RAG system's answer correctly answers a technical documentation question.

Question: {question}

Ground Truth Answer: {ground_truth}

System Answer: {answer}

Score the system answer on a scale of 0-3:
3 = Fully correct: all key facts present, nothing materially wrong
2 = Mostly correct: main point right but missing some details
1 = Partially correct: some relevant information but significant gaps or errors
0 = Incorrect: wrong, irrelevant, or "I cannot answer"

Respond with ONLY a JSON object, no other text:
{{"score": <0-3>, "reason": "one sentence explanation"}}"""


def judge_answer(client: Cerebras, question: str, ground_truth: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,        # was 200 — give more room for JSON
                temperature=0.0,
            )
            raw = response.choices[0].message.content
            if raw is None:
                raise ValueError("Empty response from model")
            raw = raw.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            # Extract JSON object even if there's extra text around it
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in: {raw[:100]}")
            data = json.loads(match.group())

            input_tokens  = response.usage.prompt_tokens     if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            return {
                "score":        min(3, max(0, int(data.get("score", 0)))),
                "reason":       data.get("reason", ""),
                "judge_tokens": input_tokens + output_tokens,
                "judge_cost":   0.0,
            }

        except Exception as e:
            if attempt == 2:
                print(f"    ✗ Judge error after 3 attempts: {e}")
                return {"score": 0, "reason": f"Judge error: {e}", "judge_tokens": 0, "judge_cost": 0.0}
            time.sleep(3)

# ── Navigation Metric ──────────────────────────────────────────────────────────

def build_target_url_lookup(dataset_path: Path) -> dict:
    if not dataset_path.exists():
        print("  Note: dataset.json not found — skipping navigated_to_target metric")
        return {}
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    questions = dataset.get("questions", dataset)
    return {q["id"]: q.get("target_url", "") for q in questions}


def check_navigation(result: dict, target_url_lookup: dict) -> bool | None:
    if not target_url_lookup:
        return None
    target_url = target_url_lookup.get(result["id"], "")
    if not target_url:
        return None
    pages_visited = result.get("pages_visited", [])
    target_norm  = target_url.rstrip("/")
    visited_norm = [p.rstrip("/") for p in pages_visited]
    return target_norm in visited_norm


# ── Statistics Helpers ─────────────────────────────────────────────────────────

def avg(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def accuracy_at_threshold(scores: list[int], threshold: int = 2) -> float:
    if not scores:
        return 0.0
    return sum(1 for s in scores if s >= threshold) / len(scores)


def compute_metrics(evaluated: list[dict], pipeline: str) -> dict:
    scores    = [r[f"{pipeline}_score"]         for r in evaluated]
    latencies = [r[f"{pipeline}_latency_s"]     for r in evaluated]
    in_toks   = [r[f"{pipeline}_input_tokens"]  for r in evaluated]
    out_toks  = [r[f"{pipeline}_output_tokens"] for r in evaluated]
    costs     = [r[f"{pipeline}_cost_usd"]      for r in evaluated]
    return {
        "n":                  len(evaluated),
        "accuracy":           round(accuracy_at_threshold(scores), 3),
        "avg_score":          round(avg(scores), 2),
        "avg_latency_s":      round(avg(latencies), 2),
        "avg_input_tokens":   round(avg(in_toks)),
        "avg_output_tokens":  round(avg(out_toks)),
        "total_cost_usd":     round(sum(costs), 4),
        "avg_cost_per_query": round(avg(costs), 6),
    }


def compute_metrics_subset(evaluated: list[dict], pipeline: str,
                            field: str, value: str) -> dict:
    subset = [r for r in evaluated if r.get(field) == value]
    if not subset:
        return {}
    return compute_metrics(subset, pipeline)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    with open(FAISS_FILE, encoding="utf-8") as f:
        faiss_results = json.load(f)
    with open(PAGEINDEX_FILE, encoding="utf-8") as f:
        pi_results_raw = json.load(f)

    # Drop errored PageIndex results
    pi_results = [r for r in pi_results_raw if "error" not in r]

    faiss_by_id = {r["id"]: r for r in faiss_results}
    pi_by_id    = {r["id"]: r for r in pi_results}

    common_ids = sorted(set(faiss_by_id.keys()) & set(pi_by_id.keys()))
    print(f"Questions in FAISS     : {len(faiss_results)}")
    print(f"Questions in PageIndex : {len(pi_results)}")
    print(f"Questions in BOTH      : {len(common_ids)}  ← evaluation set")

    target_url_lookup = build_target_url_lookup(DATASET_FILE)

    client = setup_cerebras()
    print(f"\nUsing {CEREBRAS_MODEL} as judge.")

    # Resume from existing output if present
    if OUTPUT_FULL.exists():
        with open(OUTPUT_FULL, encoding="utf-8") as f:
            evaluated = json.load(f)
        done_ids = {r["id"] for r in evaluated}
        print(f"  Resuming — {len(done_ids)} already judged, "
              f"{len(common_ids) - len(done_ids)} remaining.")
    else:
        evaluated = []
        done_ids  = set()

    for i, qid in enumerate(common_ids):
        if qid in done_ids:
            continue

        faiss_r = faiss_by_id[qid]
        pi_r    = pi_by_id[qid]

        print(f"  [{i+1}/{len(common_ids)}] {qid} [{faiss_r['difficulty']}] "
              f"{faiss_r['question'][:60]}...")

        faiss_judgment = judge_answer(client, faiss_r["question"],
                                      faiss_r["ground_truth"], faiss_r["answer"])
        time.sleep(API_DELAY)

        pi_judgment = judge_answer(client, pi_r["question"],
                                   pi_r["ground_truth"], pi_r["answer"])
        time.sleep(API_DELAY)

        navigated_to_target = check_navigation(pi_r, target_url_lookup)

        if faiss_judgment["score"] > pi_judgment["score"]:
            winner = "faiss"
        elif pi_judgment["score"] > faiss_judgment["score"]:
            winner = "pageindex"
        else:
            winner = "tie"

        record = {
            "id":         qid,
            "question":   faiss_r["question"],
            "difficulty": faiss_r["difficulty"],
            "library":    faiss_r["library"],

            "faiss_answer":        faiss_r["answer"],
            "faiss_score":         faiss_judgment["score"],
            "faiss_score_reason":  faiss_judgment["reason"],
            "faiss_latency_s":     faiss_r["latency_s"],
            "faiss_input_tokens":  faiss_r["input_tokens"],
            "faiss_output_tokens": faiss_r["output_tokens"],
            "faiss_cost_usd":      faiss_r["cost_usd"],

            "pageindex_answer":        pi_r["answer"],
            "pageindex_score":         pi_judgment["score"],
            "pageindex_score_reason":  pi_judgment["reason"],
            "pageindex_latency_s":     pi_r["latency_s"],
            "pageindex_input_tokens":  pi_r["input_tokens"],
            "pageindex_output_tokens": pi_r["output_tokens"],
            "pageindex_cost_usd":      pi_r["cost_usd"],
            "pageindex_hops":          pi_r.get("hops", 0),
            "pageindex_pages_visited": len(pi_r.get("pages_visited", [])),

            "navigated_to_target": navigated_to_target,
            "ground_truth":        faiss_r["ground_truth"],
            "winner":              winner,
        }

        evaluated.append(record)
        done_ids.add(qid)

        # Atomic save — write to temp then rename to avoid corruption on Ctrl+C
        tmp = OUTPUT_FULL.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(evaluated, f, indent=2, ensure_ascii=False)
        tmp.replace(OUTPUT_FULL)

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    faiss_overall = compute_metrics(evaluated, "faiss")
    pi_overall    = compute_metrics(evaluated, "pageindex")

    faiss_single  = compute_metrics_subset(evaluated, "faiss",     "difficulty", "single_hop")
    faiss_multi   = compute_metrics_subset(evaluated, "faiss",     "difficulty", "multi_hop")
    pi_single     = compute_metrics_subset(evaluated, "pageindex", "difficulty", "single_hop")
    pi_multi      = compute_metrics_subset(evaluated, "pageindex", "difficulty", "multi_hop")

    faiss_numpy   = compute_metrics_subset(evaluated, "faiss",     "library", "numpy")
    faiss_pandas  = compute_metrics_subset(evaluated, "faiss",     "library", "pandas")
    pi_numpy      = compute_metrics_subset(evaluated, "pageindex", "library", "numpy")
    pi_pandas     = compute_metrics_subset(evaluated, "pageindex", "library", "pandas")

    # Navigation success rate
    nav_results      = [r for r in evaluated if r["navigated_to_target"] is not None]
    nav_success_rate = (
        sum(1 for r in nav_results if r["navigated_to_target"]) / len(nav_results)
        if nav_results else None
    )

    # Accuracy split by navigation success
    nav_true  = [r for r in nav_results if r["navigated_to_target"] is True]
    nav_false = [r for r in nav_results if r["navigated_to_target"] is False]
    pi_acc_nav_true  = accuracy_at_threshold([r["pageindex_score"] for r in nav_true])
    pi_acc_nav_false = accuracy_at_threshold([r["pageindex_score"] for r in nav_false])

    faiss_wins = sum(1 for r in evaluated if r["winner"] == "faiss")
    pi_wins    = sum(1 for r in evaluated if r["winner"] == "pageindex")
    ties       = sum(1 for r in evaluated if r["winner"] == "tie")

    summary = {
        "evaluation_set_size": len(evaluated),
        "judge_model":         CEREBRAS_MODEL,
        "overall": {
            "faiss":     faiss_overall,
            "pageindex": pi_overall,
        },
        "by_difficulty": {
            "single_hop": {"faiss": faiss_single, "pageindex": pi_single},
            "multi_hop":  {"faiss": faiss_multi,  "pageindex": pi_multi},
        },
        "by_library": {
            "numpy":  {"faiss": faiss_numpy,  "pageindex": pi_numpy},
            "pandas": {"faiss": faiss_pandas, "pageindex": pi_pandas},
        },
        "winner_breakdown": {
            "faiss_wins":     faiss_wins,
            "pageindex_wins": pi_wins,
            "ties":           ties,
        },
        "pageindex_navigation": {
            "nav_success_rate":        round(nav_success_rate, 3) if nav_success_rate else "N/A",
            "questions_with_nav_data": len(nav_results),
            "accuracy_when_reached_target":     round(pi_acc_nav_true,  3),
            "accuracy_when_missed_target":      round(pi_acc_nav_false, 3),
        },
    }

    tmp = OUTPUT_SUMMARY.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    tmp.replace(OUTPUT_SUMMARY)

    # ── Print table ────────────────────────────────────────────────────────────
    def pct(x): return f"{x*100:.1f}%"

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE — PageIndex vs FAISS")
    print("=" * 65)
    print(f"  Questions evaluated  : {len(evaluated)}")
    print()
    print(f"  {'Metric':<32} {'FAISS':>10} {'PageIndex':>10}")
    print(f"  {'-'*32} {'-'*10} {'-'*10}")
    print(f"  {'Overall accuracy':<32} {pct(faiss_overall['accuracy']):>10} {pct(pi_overall['accuracy']):>10}")
    print(f"  {'Avg score (0-3)':<32} {faiss_overall['avg_score']:>10} {pi_overall['avg_score']:>10}")
    print(f"  {'Avg latency':<32} {faiss_overall['avg_latency_s']:>9}s {pi_overall['avg_latency_s']:>9}s")
    print(f"  {'Avg input tokens':<32} {faiss_overall['avg_input_tokens']:>10} {pi_overall['avg_input_tokens']:>10}")
    print(f"  {'Total cost (USD)':<32} ${faiss_overall['total_cost_usd']:>9} ${pi_overall['total_cost_usd']:>9}")
    print()
    print(f"  {'--- By Difficulty ---'}")
    print(f"  {'Single-hop accuracy':<32} {pct(faiss_single.get('accuracy',0)):>10} {pct(pi_single.get('accuracy',0)):>10}")
    print(f"  {'Multi-hop accuracy':<32} {pct(faiss_multi.get('accuracy',0)):>10} {pct(pi_multi.get('accuracy',0)):>10}")
    print()
    print(f"  {'--- By Library ---'}")
    print(f"  {'NumPy accuracy':<32} {pct(faiss_numpy.get('accuracy',0)):>10} {pct(pi_numpy.get('accuracy',0)):>10}")
    print(f"  {'Pandas accuracy':<32} {pct(faiss_pandas.get('accuracy',0)):>10} {pct(pi_pandas.get('accuracy',0)):>10}")
    print()
    print(f"  {'--- Winner Breakdown ---'}")
    print(f"  FAISS wins    : {faiss_wins}")
    print(f"  PageIndex wins: {pi_wins}")
    print(f"  Ties          : {ties}")
    if nav_success_rate is not None:
        print(f"\n  {'--- PageIndex Navigation ---'}")
        print(f"  Navigation success rate      : {pct(nav_success_rate)}")
        print(f"  Accuracy when reached target : {pct(pi_acc_nav_true)}")
        print(f"  Accuracy when missed target  : {pct(pi_acc_nav_false)}")
    print()
    acc_diff   = pi_overall['accuracy'] - faiss_overall['accuracy']
    multi_diff = pi_multi.get('accuracy', 0) - faiss_multi.get('accuracy', 0)
    print("  Resume bullet:")
    print(f"  PageIndex {pct(pi_overall['accuracy'])} vs FAISS {pct(faiss_overall['accuracy'])} overall "
          f"({'+'if acc_diff>=0 else ''}{pct(acc_diff)}).")
    print(f"  Multi-hop: PageIndex {pct(pi_multi.get('accuracy',0))} vs FAISS {pct(faiss_multi.get('accuracy',0))} "
          f"({'+'if multi_diff>=0 else ''}{pct(multi_diff)}).")
    print("=" * 65)


if __name__ == "__main__":
    main()