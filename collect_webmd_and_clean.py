"""
Collect WebMD reviews from Kaggle dataset + clean news articles HTML.
"""
import pandas as pd
import re
import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# =============================================================================
# 1. EXTRACT WEBMD REVIEWS FROM KAGGLE DATASET
# =============================================================================

def extract_webmd():
    print("[WebMD] Extracting Ozempic/Wegovy reviews from Kaggle dataset...")

    df = pd.read_csv('/tmp/webmd_kaggle/webmd.csv')

    # Filter for Ozempic/Wegovy/semaglutide
    mask = df.apply(
        lambda row: any(
            'ozempic' in str(v).lower() or 'wegovy' in str(v).lower() or 'semaglutide' in str(v).lower()
            for v in row
        ), axis=1
    )
    matched = df[mask].copy()
    print(f"  Found {len(matched)} reviews")

    # Map to expected schema
    rows = []
    for i, (_, row) in enumerate(matched.iterrows()):
        drug_raw = str(row['Drug']).lower()
        if 'ozempic' in drug_raw:
            drug = 'Ozempic'
        elif 'wegovy' in drug_raw:
            drug = 'Wegovy'
        else:
            drug = 'Semaglutide'

        # Average the three rating columns for an overall rating (1-5 scale)
        eff = row.get('Effectiveness', 3)
        sat = row.get('Satisfaction', 3)
        ease = row.get('EaseofUse', 3)
        rating = round((eff + sat + ease) / 3)

        rows.append({
            'id': f'webmd_{i}',
            'source': 'webmd',
            'drug': drug,
            'text': str(row['Reviews']),
            'rating': rating,
            'date': str(row.get('Date', '')),
            'condition': str(row.get('Condition', '')),
        })

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, 'webmd_reviews.csv')
    df_out.to_csv(out_path, index=False)
    print(f"  Saved {len(df_out)} WebMD reviews to {out_path}")
    return df_out


# =============================================================================
# 2. CLEAN NEWS ARTICLES (remove HTML from RSS descriptions)
# =============================================================================

def clean_news():
    print("\n[News] Cleaning HTML from news articles...")

    news_path = os.path.join(DATA_DIR, 'news_articles.csv')
    df = pd.read_csv(news_path)
    print(f"  Loaded {len(df)} articles")

    def clean_html(text):
        if pd.isna(text):
            return ''
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['text'] = df['text'].apply(clean_html)

    # Remove articles with very short text (just titles with no substance)
    df = df[df['text'].str.len() >= 30].reset_index(drop=True)
    df['id'] = [f'news_{i}' for i in range(len(df))]

    df.to_csv(news_path, index=False)
    print(f"  Saved {len(df)} cleaned articles to {news_path}")
    return df


if __name__ == '__main__':
    df_webmd = extract_webmd()
    df_news = clean_news()

    # Summary
    reddit_path = os.path.join(DATA_DIR, 'reddit_posts.csv')
    df_reddit = pd.read_csv(reddit_path)

    print("\n" + "=" * 50)
    print("FINAL DATA SUMMARY")
    print("=" * 50)
    print(f"Reddit posts:   {len(df_reddit)}")
    print(f"WebMD reviews:  {len(df_webmd)}")
    print(f"News articles:  {len(df_news)}")
    print(f"Total:          {len(df_reddit) + len(df_webmd) + len(df_news)}")
    print()
    print("Sample Reddit post:")
    print(f"  [{df_reddit.iloc[0]['id']}] {df_reddit.iloc[0]['text'][:150]}...")
    print()
    print("Sample WebMD review:")
    print(f"  [{df_webmd.iloc[0]['id']}] {df_webmd.iloc[0]['text'][:150]}...")
    print()
    print("Sample news article:")
    print(f"  [{df_news.iloc[0]['id']}] {df_news.iloc[0]['text'][:150]}...")
