"""
Real data collection script for INSY 669 GLP-1 Text Analytics project.
Collects from: Arctic Shift (Reddit), WebMD (scraping), Google News RSS.
"""

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from googlenewsdecoder import gnewsdecoder
except Exception:  # pragma: no cover - optional dependency at runtime.
    gnewsdecoder = None

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Date range: Jan 1, 2024 to Nov 30, 2024
AFTER_TS = int(datetime(2024, 1, 1).timestamp())
BEFORE_TS = int(datetime(2024, 11, 30).timestamp())

MIN_BODY_TOKENS_DEFAULT = 80
MAX_BODY_CHARS = 12000
MAX_FETCH_BYTES = 600000


def normalize_whitespace(text):
    """Collapse whitespace and coerce null-ish values to empty string."""
    if pd.isna(text):
        return ''
    return re.sub(r'\s+', ' ', str(text)).strip()


def clean_snippet_text(text):
    """Strip lightweight HTML/noise from RSS snippets."""
    raw = normalize_whitespace(text)
    if not raw:
        return ''
    soup = BeautifulSoup(raw, 'html.parser')
    cleaned = soup.get_text(' ', strip=True)
    return normalize_whitespace(cleaned)


def decode_news_url(rss_url):
    """
    Decode Google News RSS article URLs to direct publisher URLs when possible.
    Returns the original URL if decoder is unavailable or decoding fails.
    """
    url = normalize_whitespace(rss_url)
    if not url:
        return ''
    if 'news.google.com/rss/articles/' not in url:
        return url
    if gnewsdecoder is None:
        return url
    try:
        decoded = gnewsdecoder(url, interval=0)
    except Exception:
        return url
    if decoded.get('status') and decoded.get('decoded_url'):
        return normalize_whitespace(decoded['decoded_url'])
    return url


def extract_article_body(article_url, timeout=15):
    """
    Fetch and extract main article text using robust HTML heuristics.
    Returns empty string on failure.
    """
    url = normalize_whitespace(article_url)
    if not url or 'news.google.com/rss/articles/' in url:
        return ''

    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        resp.raise_for_status()
    except Exception:
        return ''

    chunks = []
    total = 0
    try:
        for chunk in resp.iter_content(chunk_size=16384):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= MAX_FETCH_BYTES:
                break
    except Exception:
        return ''
    finally:
        resp.close()

    try:
        html = b''.join(chunks).decode(resp.encoding or 'utf-8', errors='ignore')
    except Exception:
        return ''

    soup = BeautifulSoup(html, 'lxml')

    # Remove obvious non-content nodes before extraction.
    for tag in soup(['script', 'style', 'noscript', 'svg', 'footer', 'nav', 'aside', 'form', 'header']):
        tag.decompose()

    selectors = [
        'article',
        'main',
        '[itemprop="articleBody"]',
        '.article-body',
        '.story-body',
        '.entry-content',
        '.post-content',
    ]

    candidates = []
    for selector in selectors:
        for node in soup.select(selector):
            txt = normalize_whitespace(node.get_text(' ', strip=True))
            if len(txt) >= 300:
                candidates.append(txt)

    if not candidates:
        paragraphs = []
        for node in soup.find_all('p'):
            txt = normalize_whitespace(node.get_text(' ', strip=True))
            if len(txt) >= 40:
                paragraphs.append(txt)
        if paragraphs:
            joined = normalize_whitespace(' '.join(paragraphs))
            if len(joined) >= 300:
                candidates.append(joined)

    if not candidates:
        return ''

    best = max(candidates, key=len)
    return best[:MAX_BODY_CHARS]


# =============================================================================
# 1. REDDIT via Arctic Shift
# =============================================================================

def collect_reddit():
    """Collect Reddit posts from Arctic Shift API."""
    print("\n[1/3] Collecting Reddit posts from Arctic Shift...")

    subreddits = ['Ozempic', 'Semaglutide', 'WegovyWeightLoss']
    all_posts = []

    for sub in subreddits:
        print(f"  Fetching r/{sub}...")
        collected = 0
        before = BEFORE_TS

        while collected < 300:
            url = "https://arctic-shift.photon-reddit.com/api/posts/search"
            params = {
                'subreddit': sub,
                'after': AFTER_TS,
                'before': before,
                'limit': 100,
            }

            try:
                resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                data = resp.json().get('data', [])
            except Exception as e:
                print(f"    Error: {e}")
                break

            if not data:
                break

            for post in data:
                title = post.get('title', '') or ''
                selftext = post.get('selftext', '') or ''

                # Skip removed/deleted posts
                if selftext in ('[removed]', '[deleted]'):
                    selftext = ''

                text = f"{title}. {selftext}".strip() if selftext else title.strip()

                # Skip very short posts
                if len(text) < 20:
                    continue

                created = post.get('created_utc', 0)
                date_str = datetime.utcfromtimestamp(created).strftime('%Y-%m-%d')

                # Determine drug mentioned
                text_lower = text.lower()
                if 'ozempic' in text_lower and 'wegovy' in text_lower:
                    drug = 'Both'
                elif 'wegovy' in text_lower:
                    drug = 'Wegovy'
                elif 'ozempic' in text_lower:
                    drug = 'Ozempic'
                elif 'semaglutide' in text_lower:
                    drug = 'Semaglutide'
                else:
                    drug = 'GLP-1'

                all_posts.append({
                    'id': post.get('id', ''),
                    'source': 'reddit',
                    'subreddit': f"r/{sub}",
                    'text': text,
                    'date': date_str,
                    'score': post.get('score', 0),
                    'num_comments': post.get('num_comments', 0),
                    'drug_mentioned': drug,
                })
                collected += 1

            # Paginate: use the oldest post's timestamp
            oldest = min(p.get('created_utc', before) for p in data)
            if oldest >= before:
                break
            before = oldest

            time.sleep(0.5)

        print(f"    Collected {collected} posts from r/{sub}")

    df = pd.DataFrame(all_posts)
    # Deduplicate by post id
    df = df.drop_duplicates(subset='id')

    out_path = os.path.join(DATA_DIR, 'reddit_posts.csv')
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} Reddit posts to {out_path}")
    return df


# =============================================================================
# 2. WEBMD REVIEWS via scraping
# =============================================================================

def collect_webmd():
    """Scrape WebMD patient reviews for Ozempic and Wegovy."""
    print("\n[2/3] Scraping WebMD reviews...")

    drugs = {
        'Ozempic': 'https://reviews.webmd.com/drugs/drugreview-174491-ozempic-subcutaneous',
        'Wegovy': 'https://reviews.webmd.com/drugs/drugreview-180780-wegovy-subcutaneous',
    }

    all_reviews = []
    review_id = 0

    for drug_name, base_url in drugs.items():
        print(f"  Fetching {drug_name} reviews...")
        page = 1
        consecutive_failures = 0

        while page <= 30 and consecutive_failures < 3:
            url = f"{base_url}?page={page}&next_page=true&sort_selected=Top+Reviews&conditionFilter=-1"

            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, 'html.parser')
            except Exception as e:
                print(f"    Page {page} error: {e}")
                consecutive_failures += 1
                page += 1
                time.sleep(2)
                continue

            # Find review containers
            reviews = soup.find_all('div', class_='review-comment')
            if not reviews:
                # Try alternative selectors
                reviews = soup.find_all('p', class_='description-text')
            if not reviews:
                reviews = soup.select('[class*="review"]')

            if not reviews:
                consecutive_failures += 1
                page += 1
                time.sleep(1)
                continue

            consecutive_failures = 0

            for rev in reviews:
                text = rev.get_text(strip=True)
                if len(text) < 20:
                    continue

                # Try to find rating
                rating_el = rev.find_parent().find('div', class_=re.compile(r'rating')) if rev.find_parent() else None
                rating = ''
                if rating_el:
                    rating_text = rating_el.get_text()
                    nums = re.findall(r'(\d)', rating_text)
                    if nums:
                        rating = nums[0]

                all_reviews.append({
                    'id': f'webmd_{review_id}',
                    'source': 'webmd',
                    'drug': drug_name,
                    'text': text,
                    'rating': rating,
                    'date': '',
                    'condition': '',
                })
                review_id += 1

            page += 1
            time.sleep(1.5)

        print(f"    Collected {sum(1 for r in all_reviews if r['drug'] == drug_name)} reviews for {drug_name}")

    df = pd.DataFrame(all_reviews)

    # Fallback for upstream WebMD endpoint instability.
    # If live scraping yields nothing, recover previously collected WebMD rows
    # from the local processed corpus so the public corpus remains represented.
    if df.empty:
        fallback_path = os.path.join(DATA_DIR, 'public_processed.csv')
        if os.path.exists(fallback_path):
            try:
                fallback_df = pd.read_csv(fallback_path)
                if {'id', 'text', 'date', 'source'}.issubset(fallback_df.columns):
                    fallback_webmd = fallback_df[fallback_df['source'] == 'webmd'][['id', 'text', 'date']].copy()
                    if not fallback_webmd.empty:
                        fallback_webmd['source'] = 'webmd'
                        fallback_webmd['drug'] = 'Unknown'
                        fallback_webmd['rating'] = ''
                        fallback_webmd['condition'] = ''
                        df = fallback_webmd[['id', 'source', 'drug', 'text', 'rating', 'date', 'condition']].copy()
                        print(
                            "  [WARN] Live WebMD scraping returned 0 rows; "
                            "reused webmd rows from data/public_processed.csv fallback."
                        )
            except Exception as e:
                print(f"  [WARN] WebMD fallback load failed: {e}")

    out_path = os.path.join(DATA_DIR, 'webmd_reviews.csv')
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} WebMD reviews to {out_path}")
    return df


# =============================================================================
# 3. NEWS ARTICLES via Google News RSS + article scraping
# =============================================================================

def collect_news(fetch_body=False, body_timeout=15, min_body_tokens=MIN_BODY_TOKENS_DEFAULT):
    """
    Collect news articles about GLP-1 drugs via Google News RSS.

    Args:
        fetch_body: If True, decode Google URLs and scrape full article bodies.
        body_timeout: HTTP timeout (seconds) for body fetching.
        min_body_tokens: Minimum token threshold to treat body extraction as usable.
    """
    print("\n[3/3] Collecting news articles via Google News RSS...")
    if fetch_body:
        if gnewsdecoder is None:
            raise RuntimeError(
                "Full-body collection requested but `googlenewsdecoder` is not installed. "
                "Run: pip install googlenewsdecoder"
            )
        print("  Full-body mode enabled (decoding publisher URLs + scraping article bodies).")
    else:
        print("  Snippet-only mode enabled (title + description).")

    queries = [
        'Ozempic weight loss',
        'Wegovy weight loss',
        'semaglutide obesity',
        'GLP-1 weight loss drug',
        'Ozempic side effects',
        'Wegovy insurance coverage',
        'Ozempic shortage',
        'semaglutide clinical trial',
    ]

    all_articles = []
    seen_titles = set()
    article_id = 0
    body_success = 0
    body_attempts = 0

    for query in queries:
        print(f"  Searching: '{query}'...")
        rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}+after:2024-01-01+before:2024-12-01&hl=en-US&gl=US&ceid=US:en"

        try:
            resp = requests.get(rss_url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'xml')
        except Exception as e:
            print(f"    Error: {e}")
            continue

        items = soup.find_all('item')
        print(f"    Found {len(items)} items")

        for item in items:
            title = item.find('title')
            title_text = title.get_text(strip=True) if title else ''

            # Deduplicate
            if title_text in seen_titles or len(title_text) < 10:
                continue
            seen_titles.add(title_text)

            pub_date = item.find('pubDate')
            date_str = ''
            if pub_date:
                try:
                    dt = datetime.strptime(pub_date.text.strip()[:25], '%a, %d %b %Y %H:%M:%S')
                    date_str = dt.strftime('%Y-%m-%d')
                except:
                    date_str = ''

            source_el = item.find('source')
            source_name = source_el.get_text(strip=True) if source_el else 'Unknown'

            description = item.find('description')
            desc_html = description.get_text(' ', strip=True) if description else ''
            desc_text = clean_snippet_text(desc_html)

            link_el = item.find('link')
            rss_link = link_el.get_text(strip=True) if link_el else ''
            # Decode in a batched concurrent pass to avoid serial per-item latency.
            article_url = normalize_whitespace(rss_link)

            # Combine title and description
            text_snippet = f"{title_text}. {desc_text}" if desc_text else title_text
            text_snippet = normalize_whitespace(text_snippet)
            text_body = ''
            body_tokens = 0

            # Determine drug mentioned
            text_lower = text_snippet.lower()
            if 'ozempic' in text_lower:
                drug = 'Ozempic'
            elif 'wegovy' in text_lower:
                drug = 'Wegovy'
            elif 'semaglutide' in text_lower:
                drug = 'Semaglutide'
            else:
                drug = 'GLP-1'

            # Categorize
            cat = 'Health'
            if any(w in text_lower for w in ['fda', 'regulation', 'approval', 'lawsuit']):
                cat = 'Regulation'
            elif any(w in text_lower for w in ['stock', 'billion', 'market', 'revenue', 'sales']):
                cat = 'Business'
            elif any(w in text_lower for w in ['study', 'trial', 'research', 'clinical']):
                cat = 'Science'

            all_articles.append({
                'id': f'news_{article_id}',
                'source': source_name,
                # Keep legacy `text` as snippet for backward compatibility.
                'text': text_snippet,
                'text_snippet': text_snippet,
                'text_body': text_body,
                'body_token_count': body_tokens,
                'rss_link': rss_link,
                'article_url': article_url,
                'title': normalize_whitespace(title_text),
                'description': desc_text,
                'date': date_str,
                'drug_mentioned': drug,
                'category': cat,
            })
            article_id += 1

        time.sleep(1)

    if fetch_body and all_articles:
        print(f"  Decoding {len(all_articles)} Google RSS links concurrently...")

        def decode_job(idx, rss):
            return idx, normalize_whitespace(decode_news_url(rss))

        decoded_done = 0
        with ThreadPoolExecutor(max_workers=12) as pool:
            decode_futures = {
                pool.submit(decode_job, idx, article.get('rss_link', '')): idx
                for idx, article in enumerate(all_articles)
            }
            for fut in as_completed(decode_futures):
                idx = decode_futures[fut]
                try:
                    _, decoded_url = fut.result()
                except Exception:
                    decoded_url = normalize_whitespace(all_articles[idx].get('rss_link', ''))
                all_articles[idx]['article_url'] = decoded_url
                decoded_done += 1
                if decoded_done % 100 == 0 or decoded_done == len(all_articles):
                    print(f"    Decode progress: {decoded_done}/{len(all_articles)}")

        print(f"  Extracting article bodies concurrently for {len(all_articles)} articles...")

        def fetch_body_job(idx, url):
            if not url:
                return idx, '', 0
            body = normalize_whitespace(extract_article_body(url, timeout=body_timeout))
            tokens = len(body.split()) if body else 0
            return idx, body, tokens

        completed = 0
        max_workers = 12
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(fetch_body_job, idx, article.get('article_url', '')): idx
                for idx, article in enumerate(all_articles)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    _, text_body, body_tokens = fut.result()
                except Exception:
                    text_body, body_tokens = '', 0

                all_articles[idx]['text_body'] = text_body
                all_articles[idx]['body_token_count'] = body_tokens
                body_attempts += 1
                if body_tokens >= min_body_tokens:
                    body_success += 1

                completed += 1
                if completed % 100 == 0 or completed == len(all_articles):
                    print(
                        f"    Body extraction progress: {completed}/{len(all_articles)} "
                        f"(usable={body_success})"
                    )

    df = pd.DataFrame(all_articles)
    out_path = os.path.join(DATA_DIR, 'news_articles.csv')
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} news articles to {out_path}")
    if fetch_body:
        coverage = (body_success / body_attempts * 100.0) if body_attempts else 0.0
        print(
            f"  Full-body extraction usable for {body_success}/{body_attempts} "
            f"articles ({coverage:.1f}%, min_body_tokens={min_body_tokens})"
        )
    return df


# =============================================================================
# MAIN
# =============================================================================

def build_parser():
    parser = argparse.ArgumentParser(description="Collect Reddit, WebMD, and news data.")
    parser.add_argument(
        '--fetch-news-body',
        action='store_true',
        help='Decode Google News links and scrape full article bodies into `text_body`.',
    )
    parser.add_argument(
        '--news-body-timeout',
        type=int,
        default=15,
        help='HTTP timeout (seconds) for article body fetches.',
    )
    parser.add_argument(
        '--min-body-tokens',
        type=int,
        default=MIN_BODY_TOKENS_DEFAULT,
        help='Minimum body token threshold for extraction quality reporting.',
    )
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Starting real data collection...")
    print(f"Date range: 2024-01-01 to 2024-11-30")

    df_reddit = collect_reddit()
    df_webmd = collect_webmd()
    df_news = collect_news(
        fetch_body=args.fetch_news_body,
        body_timeout=max(5, args.news_body_timeout),
        min_body_tokens=max(10, args.min_body_tokens),
    )

    print("\n" + "=" * 50)
    print("COLLECTION SUMMARY")
    print("=" * 50)
    print(f"Reddit posts:   {len(df_reddit)}")
    print(f"WebMD reviews:  {len(df_webmd)}")
    print(f"News articles:  {len(df_news)}")
    print(f"Total:          {len(df_reddit) + len(df_webmd) + len(df_news)}")
