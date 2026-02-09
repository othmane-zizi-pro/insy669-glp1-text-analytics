#!/usr/bin/env python3
"""Run all analysis and generate figures."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from wordcloud import WordCloud
import re, os, warnings
warnings.filterwarnings('ignore')

BASE = "/home/ubuntu/classes/insy669/final-project"
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
df_reddit = pd.read_csv(f"{DATA}/reddit_posts.csv")
df_webmd = pd.read_csv(f"{DATA}/webmd_reviews.csv")
df_news = pd.read_csv(f"{DATA}/news_articles.csv")

# Combine into corpora
df_public = pd.concat([
    df_reddit[['id','text','date']].assign(source='reddit'),
    df_webmd[['id','text','date']].assign(source='webmd')
], ignore_index=True)
df_media = df_news[['id','text','date']].assign(source='news')

print(f"Public corpus: {len(df_public)} docs | Media corpus: {len(df_media)} docs")

# --- Preprocessing ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

df_public['clean'] = df_public['text'].apply(preprocess)
df_media['clean'] = df_media['text'].apply(preprocess)

# Save processed
df_public.to_csv(f"{DATA}/public_processed.csv", index=False)
df_media.to_csv(f"{DATA}/media_processed.csv", index=False)
print("Preprocessing complete")

# --- Sentiment Analysis (VADER) ---
sia = SentimentIntensityAnalyzer()

df_public['compound'] = df_public['text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
df_media['compound'] = df_media['text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

df_public['sentiment'] = df_public['compound'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))
df_media['sentiment'] = df_media['compound'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))

# Figure 1: Sentiment distribution histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_public['compound'], bins=40, color='#2196F3', alpha=0.8, edgecolor='white')
axes[0].set_title('Public Opinion Sentiment Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('VADER Compound Score')
axes[0].set_ylabel('Frequency')
axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

axes[1].hist(df_media['compound'], bins=40, color='#FF9800', alpha=0.8, edgecolor='white')
axes[1].set_title('Media Sentiment Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('VADER Compound Score')
axes[1].set_ylabel('Frequency')
axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f"{FIG}/sentiment_histograms.png", dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Box plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
data_box = pd.DataFrame({
    'Compound Score': pd.concat([df_public['compound'], df_media['compound']]),
    'Corpus': ['Public Opinion'] * len(df_public) + ['Media Coverage'] * len(df_media)
})
sns.boxplot(data=data_box, x='Corpus', y='Compound Score', palette=['#2196F3', '#FF9800'], ax=ax)
ax.set_title('Sentiment Comparison: Public vs Media', fontsize=14, fontweight='bold')
plt.savefig(f"{FIG}/sentiment_boxplot.png", dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Sentiment proportions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, (df, title, color_map) in enumerate([
    (df_public, 'Public Opinion', ['#4CAF50', '#F44336', '#9E9E9E']),
    (df_media, 'Media Coverage', ['#4CAF50', '#F44336', '#9E9E9E'])
]):
    counts = df['sentiment'].value_counts()
    axes[idx].pie(counts, labels=counts.index, autopct='%1.1f%%', colors=color_map, startangle=90)
    axes[idx].set_title(title, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIG}/sentiment_pies.png", dpi=150, bbox_inches='tight')
plt.close()

print("Sentiment analysis complete")

# --- TF-IDF Analysis ---
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=5)
tfidf_public = tfidf.fit_transform(df_public['clean'])
feature_names = tfidf.get_feature_names_out()
mean_tfidf_public = np.array(tfidf_public.mean(axis=0)).flatten()
top_public_idx = mean_tfidf_public.argsort()[-20:][::-1]
top_public_terms = [(feature_names[i], mean_tfidf_public[i]) for i in top_public_idx]

tfidf2 = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3)
tfidf_media = tfidf2.fit_transform(df_media['clean'])
feature_names2 = tfidf2.get_feature_names_out()
mean_tfidf_media = np.array(tfidf_media.mean(axis=0)).flatten()
top_media_idx = mean_tfidf_media.argsort()[-20:][::-1]
top_media_terms = [(feature_names2[i], mean_tfidf_media[i]) for i in top_media_idx]

# Figure 4: Top TF-IDF terms
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
terms_p, scores_p = zip(*top_public_terms)
axes[0].barh(range(len(terms_p)), scores_p, color='#2196F3', alpha=0.8)
axes[0].set_yticks(range(len(terms_p)))
axes[0].set_yticklabels(terms_p)
axes[0].set_title('Top TF-IDF Terms: Public', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()

terms_m, scores_m = zip(*top_media_terms)
axes[1].barh(range(len(terms_m)), scores_m, color='#FF9800', alpha=0.8)
axes[1].set_yticks(range(len(terms_m)))
axes[1].set_yticklabels(terms_m)
axes[1].set_title('Top TF-IDF Terms: Media', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(f"{FIG}/tfidf_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("TF-IDF analysis complete")

# --- Word Clouds ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
wc1 = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(' '.join(df_public['clean']))
axes[0].imshow(wc1, interpolation='bilinear')
axes[0].set_title('Public Opinion', fontsize=14, fontweight='bold')
axes[0].axis('off')

wc2 = WordCloud(width=800, height=400, background_color='white', colormap='Oranges').generate(' '.join(df_media['clean']))
axes[1].imshow(wc2, interpolation='bilinear')
axes[1].set_title('Media Coverage', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(f"{FIG}/wordclouds.png", dpi=150, bbox_inches='tight')
plt.close()
print("Word clouds complete")

# --- PMI / Lift Analysis ---
def compute_pmi_lift(texts, target_word, top_n=15, min_count=5):
    """Compute PMI and Lift for co-occurrence with target_word."""
    all_tokens = [set(doc.split()) for doc in texts]
    N = len(all_tokens)
    
    target_count = sum(1 for tokens in all_tokens if target_word in tokens)
    p_target = target_count / N
    
    word_counts = Counter()
    co_counts = Counter()
    for tokens in all_tokens:
        for w in tokens:
            word_counts[w] += 1
            if target_word in tokens and w != target_word:
                co_counts[w] += 1
    
    results = []
    for word, co_count in co_counts.items():
        if word_counts[word] < min_count:
            continue
        p_word = word_counts[word] / N
        p_co = co_count / N
        pmi = np.log2(p_co / (p_target * p_word)) if p_target * p_word > 0 else 0
        lift = p_co / (p_target * p_word) if p_target * p_word > 0 else 0
        results.append({'word': word, 'pmi': pmi, 'lift': lift, 'co_count': co_count})
    
    df_res = pd.DataFrame(results)
    if len(df_res) == 0:
        return df_res
    return df_res.sort_values('pmi', ascending=False).head(top_n)

# PMI for key terms
for target in ['ozempic', 'wegovy', 'nausea', 'weight']:
    pmi_public = compute_pmi_lift(df_public['clean'].tolist(), target)
    pmi_media = compute_pmi_lift(df_media['clean'].tolist(), target)
    
    if len(pmi_public) > 0 and len(pmi_media) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].barh(pmi_public['word'], pmi_public['pmi'], color='#2196F3', alpha=0.8)
        axes[0].set_title(f'PMI with "{target}" - Public', fontweight='bold')
        axes[0].invert_yaxis()
        
        axes[1].barh(pmi_media['word'], pmi_media['pmi'], color='#FF9800', alpha=0.8)
        axes[1].set_title(f'PMI with "{target}" - Media', fontweight='bold')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f"{FIG}/pmi_{target}.png", dpi=150, bbox_inches='tight')
        plt.close()

print("PMI analysis complete")

# --- MDS Plot ---
# Combine corpora for MDS
all_clean = pd.concat([df_public['clean'], df_media['clean']])
all_labels = ['Public'] * len(df_public) + ['Media'] * len(df_media)

# Use TF-IDF on combined, then MDS on a sample
tfidf_all = TfidfVectorizer(max_features=1000, min_df=5)
X = tfidf_all.fit_transform(all_clean)

# Sample for MDS (too many docs)
np.random.seed(42)
n_sample = 200
idx_pub = np.random.choice(len(df_public), n_sample//2, replace=False)
idx_med = np.random.choice(range(len(df_public), len(df_public)+len(df_media)), n_sample//2, replace=False)
idx_sample = np.concatenate([idx_pub, idx_med])
X_sample = X[idx_sample]
labels_sample = [all_labels[i] for i in idx_sample]

dist_matrix = 1 - cosine_similarity(X_sample)
np.fill_diagonal(dist_matrix, 0)
dist_matrix = np.maximum(dist_matrix, 0)

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=300)
coords = mds.fit_transform(dist_matrix)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#2196F3' if l == 'Public' else '#FF9800' for l in labels_sample]
ax.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.6, s=30)
ax.scatter([], [], c='#2196F3', label='Public Opinion', s=60)
ax.scatter([], [], c='#FF9800', label='Media Coverage', s=60)
ax.legend(fontsize=12)
ax.set_title('MDS Plot: Public vs Media Document Similarity', fontsize=14, fontweight='bold')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
plt.savefig(f"{FIG}/mds_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("MDS plot complete")

# --- Side Effects Analysis ---
side_effects = ['nausea', 'vomiting', 'diarrhea', 'constipation', 'headache', 'fatigue',
                'gastroparesis', 'pancreatitis', 'gallbladder', 'hair loss', 'sulfur burps',
                'stomach pain', 'anxiety', 'injection site', 'dizziness']

public_texts = ' '.join(df_public['text'].str.lower())
media_texts = ' '.join(df_media['text'].str.lower())

se_data = []
for se in side_effects:
    pub_count = public_texts.count(se)
    med_count = media_texts.count(se)
    se_data.append({'side_effect': se, 'public_mentions': pub_count, 'media_mentions': med_count})

df_se = pd.DataFrame(se_data).sort_values('public_mentions', ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(df_se))
width = 0.35
ax.barh(x - width/2, df_se['public_mentions'], width, label='Public', color='#2196F3', alpha=0.8)
ax.barh(x + width/2, df_se['media_mentions'], width, label='Media', color='#FF9800', alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(df_se['side_effect'])
ax.set_title('Side Effects: Public Mentions vs Media Coverage', fontsize=14, fontweight='bold')
ax.set_xlabel('Mention Count')
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIG}/side_effects.png", dpi=150, bbox_inches='tight')
plt.close()
print("Side effects analysis complete")

# --- Sentiment over time ---
df_public['month'] = pd.to_datetime(df_public['date']).dt.to_period('M').astype(str)
df_media['month'] = pd.to_datetime(df_media['date']).dt.to_period('M').astype(str)

pub_monthly = df_public.groupby('month')['compound'].mean()
med_monthly = df_media.groupby('month')['compound'].mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(pub_monthly.index, pub_monthly.values, 'o-', color='#2196F3', label='Public', linewidth=2)
ax.plot(med_monthly.index, med_monthly.values, 's-', color='#FF9800', label='Media', linewidth=2)
ax.set_title('Average Monthly Sentiment: Public vs Media', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Average Compound Score')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{FIG}/sentiment_timeline.png", dpi=150, bbox_inches='tight')
plt.close()
print("Timeline complete")

# --- Cosine Similarity between corpora ---
tfidf_compare = TfidfVectorizer(max_features=2000, min_df=1)
combined = [' '.join(df_public['clean']), ' '.join(df_media['clean'])]
X_compare = tfidf_compare.fit_transform(combined)
cos_sim = cosine_similarity(X_compare)[0, 1]
print(f"\nCosine similarity between Public and Media corpora: {cos_sim:.4f}")

# --- Summary stats ---
from scipy import stats
t_stat, p_value = stats.ttest_ind(df_public['compound'], df_media['compound'])
print(f"T-test: t={t_stat:.4f}, p={p_value:.6f}")
print(f"Public mean sentiment: {df_public['compound'].mean():.4f}")
print(f"Media mean sentiment: {df_media['compound'].mean():.4f}")

# Save stats
stats_dict = {
    'public_mean_sentiment': float(df_public['compound'].mean()),
    'media_mean_sentiment': float(df_media['compound'].mean()),
    't_statistic': float(t_stat),
    'p_value': float(p_value),
    'cosine_similarity': float(cos_sim),
    'public_n': len(df_public),
    'media_n': len(df_media),
    'public_sentiment_dist': df_public['sentiment'].value_counts().to_dict(),
    'media_sentiment_dist': df_media['sentiment'].value_counts().to_dict(),
}

import json
with open(f"{DATA}/analysis_stats.json", 'w') as f:
    json.dump(stats_dict, f, indent=2)

# Save processed data
df_public.to_csv(f"{DATA}/public_with_sentiment.csv", index=False)
df_media.to_csv(f"{DATA}/media_with_sentiment.csv", index=False)

print("\n=== All analysis complete! ===")
