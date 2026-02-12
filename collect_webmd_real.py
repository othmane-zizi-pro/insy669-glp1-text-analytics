"""
Scrape real WebMD patient reviews for Ozempic and Wegovy.
Uses the __INITIAL_STATE__ JSON embedded in each page.
"""
import requests
import json
import pandas as pd
import time
import os
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'en-US,en;q=0.9',
}

DRUGS = {
    'Ozempic': 'https://reviews.webmd.com/drugs/drugreview-ozempic-semaglutide',
    'Wegovy': 'https://reviews.webmd.com/drugs/drugreview-wegovy-semaglutide',
}


def extract_reviews_from_page(html):
    """Extract reviews from WebMD page HTML via __INITIAL_STATE__."""
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup.find_all('script'):
        text = script.string or ''
        if 'window.__INITIAL_STATE__' in text:
            json_str = text.replace('window.__INITIAL_STATE__=', '').strip().rstrip(';')
            data = json.loads(json_str)
            nimvs_list = data.get('all_reviews', {}).get('drug_review_nimvs', [])
            if nimvs_list:
                return nimvs_list[0].get('review_nimvs', [])
    return []


def collect_drug_reviews(drug_name, base_url):
    """Collect all reviews for a drug by paginating."""
    print(f"  Collecting {drug_name} reviews...")
    s = requests.Session()
    s.headers.update(HEADERS)

    all_reviews = []
    page = 1
    consecutive_empty = 0

    while consecutive_empty < 2:
        url = f"{base_url}?page={page}"
        try:
            r = s.get(url, timeout=15)
            if r.status_code != 200:
                print(f"    Page {page}: HTTP {r.status_code}")
                consecutive_empty += 1
                page += 1
                continue

            reviews = extract_reviews_from_page(r.text)

            if not reviews:
                consecutive_empty += 1
                page += 1
                continue

            consecutive_empty = 0

            for rev in reviews:
                text = rev.get('UserExperience', '').strip()
                if len(text) < 10:
                    continue

                all_reviews.append({
                    'drug': drug_name,
                    'text': text,
                    'date': rev.get('DatePosted', ''),
                    'effectiveness': rev.get('RatingCriteria1', ''),
                    'ease_of_use': rev.get('RatingCriteria2', ''),
                    'satisfaction': rev.get('RatingCriteria3', ''),
                    'display_name': rev.get('DisplayName', ''),
                })

            print(f"    Page {page}: {len(reviews)} reviews")
            page += 1
            time.sleep(1)

        except Exception as e:
            print(f"    Page {page} error: {e}")
            consecutive_empty += 1
            page += 1
            time.sleep(2)

    print(f"    Total: {len(all_reviews)} reviews for {drug_name}")
    return all_reviews


def main():
    print("[WebMD] Collecting real patient reviews...\n")

    all_reviews = []
    for drug_name, url in DRUGS.items():
        reviews = collect_drug_reviews(drug_name, url)
        all_reviews.extend(reviews)

    # Build DataFrame in expected schema
    rows = []
    for i, rev in enumerate(all_reviews):
        # Compute average rating (1-5 scale)
        ratings = []
        for field in ['effectiveness', 'ease_of_use', 'satisfaction']:
            try:
                r = int(rev[field])
                ratings.append(r)
            except (ValueError, TypeError):
                pass
        rating = round(sum(ratings) / len(ratings)) if ratings else ''

        # Parse date
        date_str = ''
        raw_date = rev.get('date', '')
        if raw_date:
            try:
                from datetime import datetime
                dt = datetime.strptime(raw_date.split(' ')[0], '%m/%d/%Y')
                date_str = dt.strftime('%Y-%m-%d')
            except:
                date_str = raw_date

        rows.append({
            'id': f'webmd_{i}',
            'source': 'webmd',
            'drug': rev['drug'],
            'text': rev['text'],
            'rating': rating,
            'date': date_str,
            'condition': '',
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, 'webmd_reviews.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} total WebMD reviews to {out_path}")
    print(f"  Ozempic: {len(df[df['drug']=='Ozempic'])}")
    print(f"  Wegovy: {len(df[df['drug']=='Wegovy'])}")
    return df


if __name__ == '__main__':
    main()
