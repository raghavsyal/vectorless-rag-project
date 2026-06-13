"""
scrape.py — Stage 1: Scrape NumPy and Pandas official documentation

What this script does:
- Fetches the NumPy and Pandas documentation sitemaps to discover all pages
- Visits each page and extracts: URL, title, main text content, and all internal links
- Filters out navigation/index pages that have no substantive content
- Saves the results to output/scraped_pages.json for use by parse.py

Run this first before any other script.
"""

import requests
import json
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

# Entry points for each documentation site
DOC_ROOTS = {
    "numpy": "https://numpy.org/doc/stable/",
    "pandas": "https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html",
}

# Pages that are navigation/index only — we skip these as source pages
# but may still visit them as link targets
SKIP_URL_PATTERNS = [
    "/genindex",
    "/search",
    "/py-modindex",
    "_modules/",
    "_sources/",
    "changelog",
    "whatsnew",
    "release",
    "install",
    "contributing",
]

# Minimum word count for a page to be considered "substantive"
MIN_WORD_COUNT = 100

# Delay between requests (seconds) — be polite to the servers
REQUEST_DELAY = 0.3

# Max pages to scrape per library (set to None for unlimited)
MAX_PAGES_PER_LIB = 300

OUTPUT_DIR = Path("output")
OUTPUT_FILE = OUTPUT_DIR / "scraped_pages.json"

# ── Helpers ────────────────────────────────────────────────────────────────────

def is_valid_doc_url(url: str, base_domain: str) -> bool:
    """
    Check if a URL belongs to the same documentation site and isn't
    a file download, anchor-only link, or external URL.
    """
    parsed = urlparse(url)
    # Must be same domain
    if base_domain not in parsed.netloc:
        return False
    # Skip file downloads
    if any(url.endswith(ext) for ext in [".pdf", ".zip", ".tar.gz", ".png", ".jpg", ".svg"]):
        return False
    # Skip pure anchor links
    if parsed.path == "" or parsed.path == "/":
        return False
    return True


def should_skip_page(url: str) -> bool:
    """Return True if this URL matches a known low-value page pattern."""
    return any(pattern in url for pattern in SKIP_URL_PATTERNS)


def extract_page_data(url: str, html: str, library: str) -> dict | None:
    """
    Parse HTML and extract the fields we care about:
    - title
    - body text (cleaned)
    - all internal links found in the page body
    - word count (used later to filter thin pages)
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove nav, footer, sidebar — we only want the main content
    for tag in soup.select("nav, footer, .headerlink, .sphinxsidebar, #searchbox, .related"):
        tag.decompose()

    # Get the page title
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    # Get the main content area (Sphinx docs use div.body or div[role=main])
    main = (
    soup.find("div", {"class": "body"})
    or soup.find("div", {"role": "main"})
    or soup.find("article", {"class": "bd-article"})
    or soup.find("div", {"class": "bd-article"})
    or soup.find("div", {"class": "bd-content"})
    or soup.find("article")
    or soup.find("main")
    or soup.find("body")
)

    if not main:
        return None

    body_text = main.get_text(separator=" ", strip=True)
    # Collapse whitespace
    body_text = re.sub(r"\s+", " ", body_text).strip()
    word_count = len(body_text.split())

    # Extract all <a href> links from the main content area
    base_domain = urlparse(url).netloc
    internal_links = []
    for a_tag in main.find_all("a", href=True):
        href = a_tag["href"]
        absolute = urljoin(url, href)
        # Strip fragment (anchor) from URL
        absolute = absolute.split("#")[0]
        if is_valid_doc_url(absolute, base_domain) and absolute != url:
            # Capture link text — useful later for detecting "see also" phrases
            link_text = a_tag.get_text(strip=True)
            internal_links.append({"url": absolute, "text": link_text})

    # Deduplicate links while preserving order
    seen = set()
    unique_links = []
    for link in internal_links:
        if link["url"] not in seen:
            seen.add(link["url"])
            unique_links.append(link)

    return {
        "url": url,
        "library": library,
        "title": title,
        "body_text": body_text,
        "word_count": word_count,
        "internal_links": unique_links,
    }


def crawl_library(name: str, root_url: str) -> list[dict]:
    """
    BFS crawl of a documentation site starting from root_url.
    Returns a list of page dicts.
    """
    print(f"\n── Crawling {name} docs ({root_url}) ──")

    base_domain = urlparse(root_url).netloc
    visited = set()
    queue = [root_url]
    pages = []

    session = requests.Session()
    session.headers.update({"User-Agent": "DocDatasetBuilder/1.0 (research project)"})

    while queue and (MAX_PAGES_PER_LIB is None or len(pages) < MAX_PAGES_PER_LIB):
        url = queue.pop(0)

        # Normalise URL: strip trailing slash and fragment
        url = url.rstrip("/").split("#")[0]

        if url in visited:
            continue
        visited.add(url)

        if should_skip_page(url):
            continue

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
        except Exception as e:
            print(f"  ✗ Failed to fetch {url}: {e}")
            continue

        page_data = extract_page_data(url, resp.text, name)
        if page_data is None:
            continue

        # Only keep pages with enough content to be useful
        if page_data["word_count"] >= MIN_WORD_COUNT:
            pages.append(page_data)
            if len(pages) % 25 == 0:
                print(f"  Scraped {len(pages)} pages so far...")

        # Add discovered links to the crawl queue
        for link in page_data["internal_links"]:
            link_url = link["url"].rstrip("/")
            if link_url not in visited and is_valid_doc_url(link_url, base_domain):
                queue.append(link_url)

        time.sleep(REQUEST_DELAY)

    print(f"  ✓ Done — {len(pages)} substantive pages collected from {name}")
    return pages


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    existing = json.load(open(OUTPUT_FILE, encoding="utf-8")) if OUTPUT_FILE.exists() else []
    all_pages = list(existing)

    for lib_name, root_url in DOC_ROOTS.items():
        existing_count = sum(1 for p in existing if p["library"] == lib_name)
        if existing_count >= 50:
            print(f"  Skipping {lib_name} — already scraped ({existing_count} pages)")
            continue
        # Remove stale partial scrape before re-crawling
        all_pages = [p for p in all_pages if p["library"] != lib_name]
        print(f"  Re-scraping {lib_name} — only {existing_count} pages found previously")
        pages = crawl_library(lib_name, root_url)   # ← this line was missing
        all_pages.extend(pages)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, indent=2, ensure_ascii=False)

    numpy_count  = sum(1 for p in all_pages if p["library"] == "numpy")
    pandas_count = sum(1 for p in all_pages if p["library"] == "pandas")
    total_links  = sum(len(p["internal_links"]) for p in all_pages)

    print("\n" + "=" * 55)
    print("STAGE 1 COMPLETE — Scraping Summary")
    print("=" * 55)
    print(f"  NumPy pages scraped  : {numpy_count}")
    print(f"  Pandas pages scraped : {pandas_count}")
    print(f"  Total pages          : {len(all_pages)}")
    print(f"  Total internal links : {total_links}")
    print(f"  Output saved to      : {OUTPUT_FILE}")
    print("=" * 55)
    print("Next step: run  python parse.py")


if __name__ == "__main__":
    main()
