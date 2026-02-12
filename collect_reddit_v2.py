"""
Improved Reddit collection - paginate across full date range using Arctic Shift.
Collects posts month-by-month to ensure broad date coverage.
"""
import pandas as pd
import requests
import time
import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
HEADERS = {'User-Agent': 'INSY669-TextAnalytics/1.0'}

# Monthly boundaries for 2024
MONTHS = [
    (datetime(2024, m, 1), datetime(2024, m + 1, 1) if m < 12 else datetime(2025, 1, 1))
    for m in range(1, 12)  # Jan through Nov
]


def fetch_posts(subreddit, after_ts, before_ts, limit=100):
    """Fetch posts from Arctic Shift API."""
    url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    params = {
        'subreddit': subreddit,
        'after': after_ts,
        'before': before_ts,
        'limit': limit,
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json().get('data', [])
    except Exception as e:
        print(f"      Error: {e}")
        return []


def process_post(post, subreddit):
    """Convert raw API post to our schema."""
    title = post.get('title', '') or ''
    selftext = post.get('selftext', '') or ''

    if selftext in ('[removed]', '[deleted]', ''):
        text = title.strip()
    else:
        text = f"{title}. {selftext}".strip()

    if len(text) < 20:
        return None

    created = post.get('created_utc', 0)
    date_str = datetime.utcfromtimestamp(created).strftime('%Y-%m-%d')

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

    return {
        'id': post.get('id', ''),
        'source': 'reddit',
        'subreddit': f"r/{subreddit}",
        'text': text,
        'date': date_str,
        'score': post.get('score', 0),
        'num_comments': post.get('num_comments', 0),
        'drug_mentioned': drug,
    }


def collect_reddit():
    subreddits = ['Ozempic', 'Semaglutide', 'WegovyWeightLoss']
    all_posts = []

    for sub in subreddits:
        print(f"\n  r/{sub}:")
        sub_count = 0

        for start, end in MONTHS:
            month_name = start.strftime('%b %Y')
            after_ts = int(start.timestamp())
            before_ts = int(end.timestamp())

            # Fetch up to 100 posts per month per subreddit
            data = fetch_posts(sub, after_ts, before_ts, limit=100)
            time.sleep(0.3)

            month_posts = []
            for post in data:
                processed = process_post(post, sub)
                if processed:
                    month_posts.append(processed)

            all_posts.extend(month_posts)
            sub_count += len(month_posts)
            print(f"    {month_name}: {len(month_posts)} posts")

        print(f"    Total for r/{sub}: {sub_count}")

    df = pd.DataFrame(all_posts)
    df = df.drop_duplicates(subset='id')

    out_path = os.path.join(DATA_DIR, 'reddit_posts.csv')
    df.to_csv(out_path, index=False)
    print(f"\n  Saved {len(df)} Reddit posts to {out_path}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Subreddit breakdown: {df['subreddit'].value_counts().to_dict()}")
    return df


if __name__ == '__main__':
    print("Collecting Reddit posts (month-by-month)...")
    df = collect_reddit()
