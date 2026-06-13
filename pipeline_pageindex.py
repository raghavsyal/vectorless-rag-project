"""
pipeline_pageindex.py — Pipeline B: Vectorless RAG with document tree navigation

Architecture:
- Builds a hierarchical tree from scraped_pages.json using URL path structure
- LLM navigates the tree top-down: sees current node + children + cross-ref links
- Never guesses URLs — only navigates to nodes that exist in the tree
- Follows explicit cross-reference links found in page content
- Resumes from existing output if interrupted
"""

import json
import time
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import urlparse

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

PAGES_FILE   = Path("output/scraped_pages.json")
DATASET_FILE = Path("output/dataset.json")
OUTPUT_FILE  = Path("output/results_pageindex.json")

CEREBRAS_MODEL = "gpt-oss-120b"
MAX_HOPS       = 6
PAGE_EXCERPT   = 2500   # enough to capture See Also sections
API_DELAY      = 13.0

COST_INPUT_PER_1M  = 0.0
COST_OUTPUT_PER_1M = 0.0

# ── Tree Builder ───────────────────────────────────────────────────────────────

def get_path_parts(url: str) -> list[str]:
    """Split URL path into meaningful segments for tree placement."""
    parsed = urlparse(url)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    return parts


def build_tree(pages: list[dict]) -> dict:
    """
    Build a hierarchical tree from pages using URL path structure.

    Tree node structure:
    {
        "title":    str,
        "url":      str | None,
        "library":  str | None,
        "children": {segment: node},
        "page":     dict | None   # full page data if this node has a page
    }

    Example tree path for numpy:
      root → "doc" → "stable" → "reference" → "generated" → "numpy.lexsort"
    """
    root = {"title": "root", "url": None, "library": None,
            "children": {}, "page": None}

    for page in pages:
        parts = get_path_parts(page["url"])
        node  = root
        for part in parts:
            if part not in node["children"]:
                node["children"][part] = {
                    "title":    part,
                    "url":      None,
                    "library":  None,
                    "children": {},
                    "page":     None,
                }
            node = node["children"][part]
        # Attach page data to the leaf node
        node["title"]   = page["title"]
        node["url"]     = page["url"]
        node["library"] = page["library"]
        node["page"]    = page

    return root


def get_library_root(tree: dict, library: str) -> dict | None:
    """
    Find the subtree root for a given library.
    For numpy: tree → doc → stable
    For pandas: tree → pandas-docs → stable
    """
    def search(node: dict, depth: int = 0) -> dict | None:
        # Check if this node has pages from the target library
        if node.get("library") == library:
            return node
        # Search children — find the child with most library pages
        library_counts = {}
        for key, child in node["children"].items():
            count = count_library_pages(child, library)
            if count > 0:
                library_counts[key] = count
        if not library_counts:
            return None
        # Return the child with the most library pages as the entry point
        best = max(library_counts, key=library_counts.get)
        if depth < 3:
            return search(node["children"][best], depth + 1)
        return node["children"][best]

    return search(tree)


def count_library_pages(node: dict, library: str) -> int:
    """Count how many pages in this subtree belong to the given library."""
    count = 1 if node.get("library") == library else 0
    for child in node["children"].values():
        count += count_library_pages(child, library)
    return count


def format_tree_node(node: dict, depth: int = 0, max_depth: int = 2) -> str:
    """
    Format a tree node for display to the LLM.
    Shows the node's page (if any) and its children up to max_depth.
    """
    lines = []
    indent = "  " * depth

    if node.get("url"):
        lines.append(f"{indent}[PAGE] {node['title']}")
        lines.append(f"{indent}       {node['url']}")
    else:
        lines.append(f"{indent}[DIR]  {node['title']}/")

    if depth < max_depth:
        for child in list(node["children"].values())[:20]:
            lines.append(format_tree_node(child, depth + 1, max_depth))

    return "\n".join(lines)


def get_node_by_url(tree: dict, url: str) -> dict | None:
    """Find a tree node by its URL."""
    parts = get_path_parts(url)
    node  = tree
    for part in parts:
        if part not in node["children"]:
            return None
        node = node["children"][part]
    return node if node.get("url") == url else None


def get_node_children_summary(node: dict) -> str:
    """List immediate children of a node for navigation."""
    if not node["children"]:
        return "  (no subdirectories)"
    lines = []
    for key, child in list(node["children"].items())[:30]:
        if child.get("url"):
            lines.append(f"  [PAGE] {child['title']} — {child['url']}")
        else:
            n_pages = count_library_pages(child, child.get("library", ""))
            lines.append(f"  [DIR]  {key}/ ({n_pages} pages inside)")
    return "\n".join(lines) if lines else "  (empty)"


# ── Page Formatting ────────────────────────────────────────────────────────────

def format_page_with_links(page: dict) -> str:
    """
    Format page content for the LLM.
    Shows full excerpt + explicit cross-reference links.
    2500 char excerpt ensures See Also sections are visible.
    """
    links = page.get("internal_links", [])

    # Prioritise cross-reference style links — ones with api-style text
    xref_links = [
        l for l in links
        if any(kw in l["text"].lower() for kw in
               ["see also", "refer", "numpy.", "pandas.", "scipy."])
    ]
    other_links = [l for l in links if l not in xref_links]

    link_lines = []
    for l in (xref_links + other_links)[:30]:
        link_lines.append(f"  - [{l['text']}] → {l['url']}")

    text_excerpt = page["body_text"][:PAGE_EXCERPT]
    if len(page["body_text"]) > PAGE_EXCERPT:
        text_excerpt += "... [truncated]"

    return f"""PAGE: {page['title']}
URL:  {page['url']}

CONTENT:
{text_excerpt}

CROSS-REFERENCE LINKS ON THIS PAGE:
{chr(10).join(link_lines) if link_lines else '  (none found)'}"""


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a documentation navigator. You answer technical questions by navigating a documentation tree and following cross-references.

You navigate by:
1. DESCENDING into directories to find relevant pages
2. FETCHING pages to read their content
3. FOLLOWING cross-reference links to related pages
4. ANSWERING once you have found the information

Always respond with ONLY a JSON object in exactly one of these formats:

Descend into a directory:
{"action": "descend", "path": "directory_name", "reason": "one sentence why"}

Fetch a page to read it:
{"action": "fetch", "url": "https://...", "reason": "one sentence why"}

Give your final answer:
{"action": "answer", "answer": "your 2-4 sentence answer", "pages_used": ["url1"]}

Rules:
- Respond with ONLY the JSON object. No text before or after.
- Never invent URLs — only use URLs you have seen in the tree or page links.
- If a page has cross-reference links, follow them to find complete answers.
- Give your answer only after reading the relevant page content."""


def make_tree_prompt(question: str, library: str, tree_summary: str) -> str:
    return f"""Question: {question}

You are navigating the {library.upper()} documentation tree. Here is the top-level structure:

{tree_summary}

Start by descending into the most relevant directory, or fetch a page directly if you can see it. Respond with JSON only."""


# ── Navigation Loop ────────────────────────────────────────────────────────────

def navigate_and_answer(
    question:    str,
    library:     str,
    tree:        dict,
    url_index:   dict,
    client:      OpenAI,
) -> dict:

    # Find the library subtree root
    lib_root = get_library_root(tree, library)
    if lib_root is None:
        lib_root = tree

    # Build initial tree summary (top 2 levels of library subtree)
    tree_summary = format_tree_node(lib_root, depth=0, max_depth=2)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": make_tree_prompt(question, library, tree_summary)},
    ]

    current_node        = lib_root
    pages_visited       = []
    total_input_tokens  = 0
    total_output_tokens = 0
    t0 = time.time()

    for hop in range(MAX_HOPS):
        raw = None

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=CEREBRAS_MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content
                if raw is None:
                    raise ValueError("Empty response")
                raw = raw.strip()
                total_input_tokens  += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens
                break
            except Exception as e:
                if attempt == 2:
                    return _error_result(str(e), pages_visited, hop,
                                         total_input_tokens, total_output_tokens, t0)
                print(f"    Retry {attempt+1}/3: {e}")
                time.sleep(10)

        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        # Extract JSON
        try:
            # Try direct parse first
            action = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON object from response
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                try:
                    action = json.loads(match.group())
                except json.JSONDecodeError:
                    return _build_result(raw[:500], pages_visited, [],
                                          hop + 1, total_input_tokens,
                                          total_output_tokens, t0)
            else:
                return _build_result(raw[:500], pages_visited, [],
                                      hop + 1, total_input_tokens,
                                      total_output_tokens, t0)

        # ── answer ────────────────────────────────────────────────────────────
        if action.get("action") == "answer":
            return _build_result(
                action.get("answer", ""),
                pages_visited,
                action.get("pages_used", []),
                hop + 1,
                total_input_tokens,
                total_output_tokens,
                t0,
            )

        # ── descend ───────────────────────────────────────────────────────────
        elif action.get("action") == "descend":
            path = action.get("path", "").strip("/")
            messages.append({"role": "assistant", "content": raw})

            # Find the child node
            target_node = None
            for key, child in current_node["children"].items():
                if key == path or path.lower() in key.lower():
                    target_node = child
                    break

            if target_node is None:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Directory '{path}' not found. "
                        f"Available directories/pages:\n"
                        f"{get_node_children_summary(current_node)}\n\n"
                        f"Try a different path or fetch a page directly."
                    )
                })
                continue

            current_node = target_node
            children_summary = get_node_children_summary(current_node)

            # If this node itself has a page, show it too
            page_preview = ""
            if current_node.get("page"):
                page = current_node["page"]
                page_preview = f"\nThis directory has a page:\n  [PAGE] {page['title']} — {page['url']}\n"

            messages.append({
                "role": "user",
                "content": (
                    f"You descended into: {path}\n"
                    f"{page_preview}"
                    f"\nContents:\n{children_summary}\n\n"
                    f"Fetch a page to read it, descend further, or give your answer."
                )
            })

        # ── fetch ─────────────────────────────────────────────────────────────
        elif action.get("action") == "fetch":
            url  = action.get("url", "").strip()
            page = url_index.get(url)
            messages.append({"role": "assistant", "content": raw})

            if page is None:
                # Try partial URL match
                for stored_url, stored_page in url_index.items():
                    if url in stored_url or stored_url.endswith(url.lstrip("/")):
                        page = stored_page
                        break

            if page is None:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Page not found: {url}\n"
                        f"Try fetching a URL you have seen in the tree or in a page's cross-reference links."
                    )
                })
                continue

            pages_visited.append(page["url"])
            messages.append({
                "role": "user",
                "content": (
                    f"Here is the page:\n\n"
                    f"{format_page_with_links(page)}\n\n"
                    f"If this page has cross-reference links relevant to the question, "
                    f"fetch one to follow it. Otherwise give your final answer."
                )
            })

        # ── unexpected ────────────────────────────────────────────────────────
        else:
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    'Respond with one of: '
                    '{"action":"descend","path":"...","reason":"..."} or '
                    '{"action":"fetch","url":"...","reason":"..."} or '
                    '{"action":"answer","answer":"...","pages_used":[]}'
                )
            })

    # ── max hops — force answer ────────────────────────────────────────────────
    messages.append({
        "role": "user",
        "content": 'Maximum navigation steps reached. Give your best answer now: {"action":"answer","answer":"...","pages_used":[]}'
    })

    try:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=CEREBRAS_MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content
                if raw is None:
                    raise ValueError("Empty response")
                raw = raw.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                total_input_tokens  += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(10)

        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        action = json.loads(match.group() if match else raw)
        answer     = action.get("answer", raw[:500])
        pages_used = action.get("pages_used", [])
    except Exception:
        answer     = "Could not extract answer after max hops."
        pages_used = []

    return _build_result(answer, pages_visited, pages_used,
                          MAX_HOPS, total_input_tokens, total_output_tokens, t0)


# ── Result Helpers ─────────────────────────────────────────────────────────────

def _calc_cost(input_tokens, output_tokens):
    return round(
        input_tokens  / 1_000_000 * COST_INPUT_PER_1M +
        output_tokens / 1_000_000 * COST_OUTPUT_PER_1M, 6)

def _build_result(answer, pages_visited, pages_used, hops,
                   input_tokens, output_tokens, t0):
    return {
        "answer":        answer,
        "pages_visited": pages_visited,
        "pages_used":    pages_used,
        "hops":          hops,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "latency_s":     round(time.time() - t0, 3),
        "cost_usd":      _calc_cost(input_tokens, output_tokens),
    }

def _error_result(error, pages_visited, hops, input_tokens, output_tokens, t0):
    return {
        "answer":        f"API error: {error}",
        "pages_visited": pages_visited,
        "pages_used":    [],
        "hops":          hops,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "latency_s":     round(time.time() - t0, 3),
        "cost_usd":      _calc_cost(input_tokens, output_tokens),
        "error":         error,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    with open(PAGES_FILE, encoding="utf-8") as f:
        pages = json.load(f)
    with open(DATASET_FILE, encoding="utf-8") as f:
        dataset = json.load(f)
        questions = dataset["questions"]

    print(f"Loaded {len(pages)} pages, {len(questions)} questions.")
    print(f"Model: {CEREBRAS_MODEL} | Max hops: {MAX_HOPS} | Excerpt: {PAGE_EXCERPT} chars")

    # Build tree and URL index
    print("Building document tree...")
    tree      = build_tree(pages)
    url_index = {p["url"]: p for p in pages}

    numpy_count  = count_library_pages(tree, "numpy")
    pandas_count = count_library_pages(tree, "pandas")
    print(f"  Tree built — numpy: {numpy_count} pages, pandas: {pandas_count} pages")

    # Resume logic
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            results = json.load(f)
        results  = [r for r in results if "error" not in r]
        done_ids = {r["id"] for r in results}
        print(f"  Resuming — {len(done_ids)} done, {len(questions)-len(done_ids)} remaining.")
    else:
        results  = []
        done_ids = set()

    client = OpenAI(
        api_key=os.getenv("CEREBRAS_API_KEY"),
        base_url="https://api.cerebras.ai/v1",
    )

    for i, q in enumerate(questions):
        if q["id"] in done_ids:
            continue

        print(f"  [{i+1}/{len(questions)}] [{q['library']}] {q['question'][:75]}...")

        result = navigate_and_answer(
            question  = q["question"],
            library   = q["library"],
            tree      = tree,
            url_index = url_index,
            client    = client,
        )

        results.append({
            "id":           q["id"],
            "question":     q["question"],
            "ground_truth": q["ground_truth_answer"],
            "library":      q["library"],
            "difficulty":   q["difficulty"],
            "pipeline":     "pageindex",
            **result,
        })

        # Atomic save
        tmp = OUTPUT_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        tmp.replace(OUTPUT_FILE)

        time.sleep(API_DELAY)

    total_cost  = sum(r.get("cost_usd", 0) for r in results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    avg_hops    = sum(r.get("hops", 0) for r in results) / len(results)
    errors      = sum(1 for r in results if "error" in r)

    print("\n" + "=" * 55)
    print("PAGEINDEX PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Questions answered : {len(results)}")
    print(f"  Avg latency        : {avg_latency:.2f}s")
    print(f"  Avg hops           : {avg_hops:.2f}")
    print(f"  Errors             : {errors}")
    print(f"  Total cost (USD)   : ${total_cost:.4f}")
    print(f"  Output saved to    : {OUTPUT_FILE}")
    print("=" * 55)
    print("Next step: run  python evaluate.py")


if __name__ == "__main__":
    main()