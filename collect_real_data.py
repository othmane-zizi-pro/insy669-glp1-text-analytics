"""
Real data collection script for INSY 669 GLP-1 Text Analytics project.
Collects from: Arctic Shift (Reddit), WebMD (scraping), Google News RSS (scraping)
No API keys required.
"""

import pandas as pd
import requests
import time
import re
import json
from datetime import datetime
from bs4 import BeautifulSoup
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Date range: Jan 1, 2024 to Nov 30, 2024
AFTER_TS = int(datetime(2024, 1, 1).timestamp())
BEFORE_TS = int(datetime(2024, 11, 30).timestamp())


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
    out_path = os.path.join(DATA_DIR, 'webmd_reviews.csv')
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} WebMD reviews to {out_path}")
    return df


# =============================================================================
# 3. NEWS ARTICLES via Google News RSS + article scraping
# =============================================================================

def collect_news():
    """Collect news articles about GLP-1 drugs via Google News RSS."""
    print("\n[3/3] Collecting news articles via Google News RSS...")

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
            desc_text = description.get_text(strip=True) if description else ''

            # Combine title and description
            text = f"{title_text}. {desc_text}" if desc_text else title_text

            # Determine drug mentioned
            text_lower = text.lower()
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
                'text': text,
                'date': date_str,
                'drug_mentioned': drug,
                'category': cat,
            })
            article_id += 1

        time.sleep(1)

    df = pd.DataFrame(all_articles)
    out_path = os.path.join(DATA_DIR, 'news_articles.csv')
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} news articles to {out_path}")
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Starting real data collection...")
    print(f"Date range: 2024-01-01 to 2024-11-30")

    df_reddit = collect_reddit()
    df_webmd = collect_webmd()
    df_news = collect_news()

    print("\n" + "=" * 50)
    print("COLLECTION SUMMARY")
    print("=" * 50)
    print(f"Reddit posts:   {len(df_reddit)}")
    print(f"WebMD reviews:  {len(df_webmd)}")
    print(f"News articles:  {len(df_news)}")
    print(f"Total:          {len(df_reddit) + len(df_webmd) + len(df_news)}")
