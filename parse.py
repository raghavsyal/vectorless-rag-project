"""
parse.py — Stage 2: Identify cross-reference pairs from scraped pages

What this script does:
- Reads output/scraped_pages.json produced by scrape.py
- Finds links that are explicit cross-references: "see also", "refer to",
  "as described in", etc. — these are signals that the linked page is
  conceptually required to fully understand the source page
- Also captures "See Also" sections which Sphinx/NumPy docs use heavily
- Filters out pairs where the target page is thin (navigation/index pages)
- Classifies each pair as single_hop or multi_hop
- Saves output/cross_ref_pairs.json for use by generate.py

Run this after scrape.py.
"""

import json
import re
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_FILE  = Path("output/scraped_pages.json")
OUTPUT_FILE = Path("output/cross_ref_pairs.json")

# Phrases in link-surrounding text that signal an explicit cross-reference.
# We look at the sentence/clause containing the link.
CROSSREF_PHRASES = [
    "see also",
    "see:",
    "refer to",
    "as described in",
    "as discussed in",
    "for more information",
    "for details",
    "for more details",
    "more information can be found",
    "further reading",
    "related:",
    "note:",          # often precedes a cross-reference in NumPy docs
    "warning:",
    "deprecated",
    "equivalent to",
    "similar to",
    "compare with",
]

# Link anchor text patterns that signal the link IS the cross-reference
# (e.g., a link whose text is "numpy.broadcast" in a See Also block)
FUNC_LINK_PATTERN = re.compile(r"^(numpy|pandas|np|pd)\.", re.IGNORECASE)

# Minimum word count for a TARGET page to be considered substantive
MIN_TARGET_WORDS = 150

# ── Helpers ────────────────────────────────────────────────────────────────────

def build_url_index(pages: list[dict]) -> dict[str, dict]:
    """Build a dict mapping URL → page data for fast lookups."""
    # Normalise URLs: strip trailing slash
    return {p["url"].rstrip("/"): p for p in pages}


def get_surrounding_text(body_text: str, link_text: str, window: int = 200) -> str:
    """
    Find link_text in body_text and return the surrounding window of characters.
    Used to check whether cross-reference phrases appear near the link.
    Returns empty string if link_text not found.
    """
    if not link_text or len(link_text) < 2:
        return ""
    idx = body_text.lower().find(link_text.lower())
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end   = min(len(body_text), idx + len(link_text) + window)
    return body_text[start:end].lower()


def is_crossref_link(link: dict, body_text: str, page_url: str) -> bool:
    """
    Decide whether a given link from a page qualifies as a cross-reference.

    Two signals:
    1. The surrounding text contains a cross-reference phrase
    2. The link text looks like a function/class name (numpy.X, pandas.X)
       and appears in a See Also block (common in NumPy/Pandas API docs)
    """
    link_text    = link.get("text", "")
    surrounding  = get_surrounding_text(body_text, link_text)

    # Signal 1: explicit cross-reference phrase near the link
    if any(phrase in surrounding for phrase in CROSSREF_PHRASES):
        return True

    # Signal 2: link text is a numpy/pandas symbol (likely in a See Also block)
    if FUNC_LINK_PATTERN.match(link_text.strip()):
        return True

    # Signal 3: check if "see also" appears anywhere on the page body
    # and the link is a same-library API reference
    if "see also" in body_text.lower() and FUNC_LINK_PATTERN.match(link_text.strip()):
        return True

    return False


def classify_hop(source_url: str, target_url: str, url_index: dict) -> str:
    """
    Classify a pair as single_hop or multi_hop.

    single_hop: the target page itself has no onward cross-references
    multi_hop:  the target page also has outbound cross-references,
                meaning a complete answer requires chaining ≥2 hops
    """
    target_page = url_index.get(target_url.rstrip("/"))
    if target_page is None:
        return "single_hop"

    target_body = target_page["body_text"]
    target_links = target_page["internal_links"]

    # Check whether the target page itself has cross-reference links
    for link in target_links:
        if is_crossref_link(link, target_body, target_url):
            return "multi_hop"

    return "single_hop"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load scraped pages
    with open(INPUT_FILE, encoding="utf-8") as f:
        pages = json.load(f)

    print(f"Loaded {len(pages)} scraped pages.")

    url_index = build_url_index(pages)
    pairs = []

    for page in pages:
        source_url  = page["url"].rstrip("/")
        body_text   = page["body_text"]
        library     = page["library"]

        for link in page["internal_links"]:
            target_url = link["url"].rstrip("/")

            # Skip self-links
            if target_url == source_url:
                continue

            # Skip if we don't have the target page in our corpus
            target_page = url_index.get(target_url)
            if target_page is None:
                continue

            # Skip thin target pages — they're likely index/nav pages
            if target_page["word_count"] < MIN_TARGET_WORDS:
                continue

            # Skip if the target is a different library
            # (we want within-library cross-references for coherence)
            if target_page["library"] != library:
                continue

            # Check if this link qualifies as a cross-reference
            if not is_crossref_link(link, body_text, source_url):
                continue

            hop_type = classify_hop(source_url, target_url, url_index)

            pairs.append({
                "library":          library,
                "source_url":       source_url,
                "source_title":     page["title"],
                "source_text":      body_text[:3000],   # truncate for later use in prompts
                "target_url":       target_url,
                "target_title":     target_page["title"],
                "target_text":      target_page["body_text"][:3000],
                "link_text":        link.get("text", ""),
                "difficulty":       hop_type,
            })

    # Deduplicate: same (source_url, target_url) pair can appear multiple times
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = (p["source_url"], p["target_url"])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_pairs, f, indent=2, ensure_ascii=False)

    # ── Stage 2 Summary ──────────────────────────────────────────────────────
    numpy_pairs  = [p for p in unique_pairs if p["library"] == "numpy"]
    pandas_pairs = [p for p in unique_pairs if p["library"] == "pandas"]
    single_hop   = [p for p in unique_pairs if p["difficulty"] == "single_hop"]
    multi_hop    = [p for p in unique_pairs if p["difficulty"] == "multi_hop"]

    print("\n" + "=" * 55)
    print("STAGE 2 COMPLETE — Cross-Reference Parsing Summary")
    print("=" * 55)
    print(f"  Total pairs found    : {len(unique_pairs)}")
    print(f"  NumPy pairs          : {len(numpy_pairs)}")
    print(f"  Pandas pairs         : {len(pandas_pairs)}")
    print(f"  Single-hop pairs     : {len(single_hop)}")
    print(f"  Multi-hop pairs      : {len(multi_hop)}")
    print(f"  Output saved to      : {OUTPUT_FILE}")
    print("=" * 55)
    print("Next step: run  python generate.py")


if __name__ == "__main__":
    main()
