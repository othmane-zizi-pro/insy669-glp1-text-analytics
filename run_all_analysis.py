"""
Run all analysis notebooks (02-07) as Python scripts to regenerate
processed data, figures, and statistics with the new real data.
"""
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import json
import warnings
import os
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

DATA_DIR = 'data'
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'


# =============================================================================
# NOTEBOOK 02: PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("NOTEBOOK 02: PREPROCESSING")
print("=" * 60)

df_reddit = pd.read_csv(f'{DATA_DIR}/reddit_posts.csv')
df_webmd = pd.read_csv(f'{DATA_DIR}/webmd_reviews.csv')
df_news = pd.read_csv(f'{DATA_DIR}/news_articles.csv')

# Create unified corpora
df_public = pd.concat([
    df_reddit[['id', 'text', 'date']].assign(source='reddit'),
    df_webmd[['id', 'text', 'date']].assign(source='webmd')
], ignore_index=True)

df_media = df_news[['id', 'text', 'date']].assign(source='news')

print(f"Public corpus: {len(df_public)} documents")
print(f"Media corpus: {len(df_media)} documents")

# Preprocessing pipeline
stop_words = set(stopwords.words('english'))
stop_words.update(['mg', 'would', 'also', 'get', 'got', 'one', 'like', 'even', 'im', 'ive'])
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


df_public['clean'] = df_public['text'].apply(preprocess)
df_media['clean'] = df_media['text'].apply(preprocess)

df_public['token_count'] = df_public['clean'].apply(lambda x: len(x.split()))
df_media['token_count'] = df_media['clean'].apply(lambda x: len(x.split()))

print(f"Public - Mean tokens: {df_public['token_count'].mean():.1f}")
print(f"Media  - Mean tokens: {df_media['token_count'].mean():.1f}")

# Save processed data
df_public.to_csv(f'{DATA_DIR}/public_processed.csv', index=False)
df_media.to_csv(f'{DATA_DIR}/media_processed.csv', index=False)
print("Processed data saved.")


# =============================================================================
# NOTEBOOK 03: SENTIMENT ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("NOTEBOOK 03: SENTIMENT ANALYSIS")
print("=" * 60)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

df_public['sentiment'] = df_public['text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
df_media['sentiment'] = df_media['text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])


def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    return 'neutral'


df_public['sentiment_label'] = df_public['sentiment'].apply(classify_sentiment)
df_media['sentiment_label'] = df_media['sentiment'].apply(classify_sentiment)

# Save with sentiment
df_public.to_csv(f'{DATA_DIR}/public_with_sentiment.csv', index=False)
df_media.to_csv(f'{DATA_DIR}/media_with_sentiment.csv', index=False)

# Statistical tests
t_stat, p_val = stats.ttest_ind(df_public['sentiment'], df_media['sentiment'])
cohens_d = (df_public['sentiment'].mean() - df_media['sentiment'].mean()) / \
           np.sqrt((df_public['sentiment'].std() ** 2 + df_media['sentiment'].std() ** 2) / 2)

print(f"Public mean sentiment: {df_public['sentiment'].mean():.4f}")
print(f"Media mean sentiment:  {df_media['sentiment'].mean():.4f}")
print(f"T-statistic: {t_stat:.3f}, P-value: {p_val:.6f}")
print(f"Cohen's d: {cohens_d:.3f}")

pub_dist = df_public['sentiment_label'].value_counts().to_dict()
med_dist = df_media['sentiment_label'].value_counts().to_dict()
print(f"Public distribution: {pub_dist}")
print(f"Media distribution: {med_dist}")

# Figure: Sentiment Histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_public['sentiment'], bins=40, color='#2196F3', alpha=0.8, edgecolor='white')
axes[0].axvline(df_public['sentiment'].mean(), color='red', linestyle='--', label=f"Mean: {df_public['sentiment'].mean():.3f}")
axes[0].set_title('Public Corpus: Sentiment Distribution', fontweight='bold')
axes[0].set_xlabel('VADER Compound Score')
axes[0].legend()
axes[1].hist(df_media['sentiment'], bins=40, color='#FF9800', alpha=0.8, edgecolor='white')
axes[1].axvline(df_media['sentiment'].mean(), color='red', linestyle='--', label=f"Mean: {df_media['sentiment'].mean():.3f}")
axes[1].set_title('Media Corpus: Sentiment Distribution', fontweight='bold')
axes[1].set_xlabel('VADER Compound Score')
axes[1].legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/sentiment_histograms.png')
plt.close()
print("Saved sentiment_histograms.png")

# Figure: Sentiment Boxplot
fig, ax = plt.subplots(figsize=(10, 6))
data = [df_public['sentiment'], df_media['sentiment']]
bp = ax.boxplot(data, labels=['Public\n(Reddit + WebMD)', 'Media\n(News)'],
                patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#2196F3')
bp['boxes'][1].set_facecolor('#FF9800')
for box in bp['boxes']:
    box.set_alpha(0.7)
ax.set_ylabel('VADER Compound Score')
ax.set_title('Sentiment Comparison: Public vs Media', fontweight='bold')
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.annotate(f'p = {p_val:.2e}', xy=(1.5, max(df_public['sentiment'].max(), df_media['sentiment'].max())),
            fontsize=12, ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/sentiment_boxplot.png')
plt.close()
print("Saved sentiment_boxplot.png")

# Figure: Sentiment Pie Charts
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = {'positive': '#4CAF50', 'neutral': '#9E9E9E', 'negative': '#F44336'}
for ax, df, title in [(axes[0], df_public, 'Public Corpus'), (axes[1], df_media, 'Media Corpus')]:
    counts = df['sentiment_label'].value_counts()
    labels = [f"{l}\n({c}, {c / len(df) * 100:.1f}%)" for l, c in counts.items()]
    ax.pie(counts, labels=labels, colors=[colors[l] for l in counts.index],
           autopct='', startangle=90, textprops={'fontsize': 11})
    ax.set_title(f'{title} (n={len(df)})', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/sentiment_pies.png')
plt.close()
print("Saved sentiment_pies.png")


# =============================================================================
# NOTEBOOK 04: ASSOCIATIONS (PMI & MDS)
# =============================================================================
print("\n" + "=" * 60)
print("NOTEBOOK 04: ASSOCIATIONS")
print("=" * 60)

# Build combined vocabulary
all_clean = pd.concat([df_public['clean'], df_media['clean']])
vectorizer = CountVectorizer(max_features=2000, min_df=5)
bow = vectorizer.fit_transform(all_clean)
vocab = vectorizer.get_feature_names_out()
word_freq = np.array(bow.sum(axis=0)).flatten()
total_words = word_freq.sum()


def compute_pmi(target_word, top_n=15):
    if target_word not in vectorizer.vocabulary_:
        return []
    target_idx = vectorizer.vocabulary_[target_word]
    target_col = bow[:, target_idx].toarray().flatten()
    p_target = target_col.sum() / bow.shape[0]
    results = []
    for i, word in enumerate(vocab):
        if word == target_word:
            continue
        word_col = bow[:, i].toarray().flatten()
        p_word = word_col.sum() / bow.shape[0]
        p_joint = ((target_col > 0) & (word_col > 0)).sum() / bow.shape[0]
        if p_joint > 0 and p_word > 0:
            pmi = np.log2(p_joint / (p_target * p_word))
            if pmi > 0:
                results.append((word, pmi))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


for target in ['ozempic', 'wegovy', 'weight']:
    pmi_results = compute_pmi(target)
    if pmi_results:
        words, scores = zip(*pmi_results)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(words)), scores, color='#2196F3', alpha=0.8)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('PMI Score')
        ax.set_title(f'Top PMI Associations: "{target}"', fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/pmi_{target}.png')
        plt.close()
        print(f"Saved pmi_{target}.png")

# MDS Plot
tfidf_all = TfidfVectorizer(max_features=3000, min_df=3).fit_transform(all_clean)
# Sample for MDS (too slow with full dataset)
n_pub = min(300, len(df_public))
n_med = min(300, len(df_media))
np.random.seed(42)
pub_idx = np.random.choice(len(df_public), n_pub, replace=False)
med_idx = np.random.choice(len(df_media), n_med, replace=False) + len(df_public)
sample_idx = np.concatenate([pub_idx, med_idx])
tfidf_sample = tfidf_all[sample_idx]
cos_sim = cosine_similarity(tfidf_sample)
cos_dist = 1 - cos_sim
np.fill_diagonal(cos_dist, 0)
cos_dist = np.maximum(cos_dist, 0)

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=300)
coords = mds.fit_transform(cos_dist)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(coords[:n_pub, 0], coords[:n_pub, 1], c='#2196F3', alpha=0.4, s=20, label='Public')
ax.scatter(coords[n_pub:, 0], coords[n_pub:, 1], c='#FF9800', alpha=0.4, s=20, label='Media')
ax.set_title('MDS: Public vs Media Document Similarity', fontweight='bold')
ax.legend(fontsize=12)
ax.set_xlabel('MDS Dimension 1')
ax.set_ylabel('MDS Dimension 2')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/mds_plot.png')
plt.close()
print(f"Saved mds_plot.png (stress: {mds.stress_:.3f})")


# =============================================================================
# NOTEBOOK 05: COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("NOTEBOOK 05: COMPARISON")
print("=" * 60)

# TF-IDF Comparison
tfidf_pub = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
tfidf_med = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)

pub_matrix = tfidf_pub.fit_transform(df_public['clean'])
med_matrix = tfidf_med.fit_transform(df_media['clean'])

pub_means = np.array(pub_matrix.mean(axis=0)).flatten()
med_means = np.array(med_matrix.mean(axis=0)).flatten()

pub_top_idx = pub_means.argsort()[-20:][::-1]
med_top_idx = med_means.argsort()[-20:][::-1]

pub_terms = [(tfidf_pub.get_feature_names_out()[i], pub_means[i]) for i in pub_top_idx]
med_terms = [(tfidf_med.get_feature_names_out()[i], med_means[i]) for i in med_top_idx]

print("Top Public TF-IDF terms:", [t[0] for t in pub_terms[:10]])
print("Top Media TF-IDF terms:", [t[0] for t in med_terms[:10]])

# Figure: TF-IDF Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
words_p, scores_p = zip(*pub_terms)
words_m, scores_m = zip(*med_terms)
axes[0].barh(range(20), scores_p, color='#2196F3', alpha=0.8)
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(words_p)
axes[0].set_title('Public: Top TF-IDF Terms', fontweight='bold')
axes[0].invert_yaxis()
axes[1].barh(range(20), scores_m, color='#FF9800', alpha=0.8)
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(words_m)
axes[1].set_title('Media: Top TF-IDF Terms', fontweight='bold')
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/tfidf_comparison.png')
plt.close()
print("Saved tfidf_comparison.png")

# Word Clouds
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
pub_text = ' '.join(df_public['clean'].dropna())
med_text = ' '.join(df_media['clean'].dropna())
wc_pub = WordCloud(width=800, height=400, background_color='white', colormap='Blues', max_words=100).generate(pub_text)
wc_med = WordCloud(width=800, height=400, background_color='white', colormap='Oranges', max_words=100).generate(med_text)
axes[0].imshow(wc_pub, interpolation='bilinear')
axes[0].set_title('Public Corpus', fontweight='bold', fontsize=16)
axes[0].axis('off')
axes[1].imshow(wc_med, interpolation='bilinear')
axes[1].set_title('Media Corpus', fontweight='bold', fontsize=16)
axes[1].axis('off')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/wordclouds.png')
plt.close()
print("Saved wordclouds.png")

# Side Effects Gap Analysis
side_effects = ['nausea', 'vomiting', 'diarrhea', 'constipation', 'headache',
                'fatigue', 'dizziness', 'bloating', 'hair loss', 'pancreatitis',
                'gastroparesis', 'depression', 'anxiety', 'insomnia', 'sulfur burp']

pub_text_lower = pub_text.lower()
med_text_lower = med_text.lower()

se_data = []
for se in side_effects:
    pub_count = pub_text_lower.count(se)
    med_count = med_text_lower.count(se)
    se_data.append({'side_effect': se, 'public': pub_count, 'media': med_count})

df_se = pd.DataFrame(se_data).sort_values('public', ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
x = range(len(df_se))
width = 0.35
ax.bar([i - width / 2 for i in x], df_se['public'], width, label='Public', color='#2196F3', alpha=0.8)
ax.bar([i + width / 2 for i in x], df_se['media'], width, label='Media', color='#FF9800', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(df_se['side_effect'], rotation=45, ha='right')
ax.set_ylabel('Mention Count')
ax.set_title('Side Effects Coverage: Public vs Media', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/side_effects.png')
plt.close()
print("Saved side_effects.png")

# Temporal Sentiment
df_public['month'] = pd.to_datetime(df_public['date'], errors='coerce').dt.to_period('M')
df_media['month'] = pd.to_datetime(df_media['date'], errors='coerce').dt.to_period('M')

pub_monthly = df_public.groupby('month')['sentiment'].mean()
med_monthly = df_media.groupby('month')['sentiment'].mean()

fig, ax = plt.subplots(figsize=(12, 6))
if len(pub_monthly) > 1:
    ax.plot(pub_monthly.index.astype(str), pub_monthly.values, 'o-', color='#2196F3', linewidth=2, label='Public')
if len(med_monthly) > 1:
    ax.plot(med_monthly.index.astype(str), med_monthly.values, 's-', color='#FF9800', linewidth=2, label='Media')
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Month')
ax.set_ylabel('Mean VADER Sentiment')
ax.set_title('Monthly Sentiment Trends: Public vs Media', fontweight='bold')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/sentiment_timeline.png')
plt.close()
print("Saved sentiment_timeline.png")

# Cosine similarity between corpora
combined_tfidf = TfidfVectorizer(max_features=5000, min_df=3)
combined_matrix = combined_tfidf.fit_transform(pd.concat([df_public['clean'], df_media['clean']]))
pub_vec = np.asarray(combined_matrix[:len(df_public)].mean(axis=0))
med_vec = np.asarray(combined_matrix[len(df_public):].mean(axis=0))
cos_sim_score = cosine_similarity(pub_vec, med_vec)[0][0]
print(f"Cosine similarity between corpora: {cos_sim_score:.3f}")


# =============================================================================
# NOTEBOOK 06: CLASSIFICATION
# =============================================================================
print("\n" + "=" * 60)
print("NOTEBOOK 06: CLASSIFICATION")
print("=" * 60)

# Prepare data
all_texts = pd.concat([df_public['clean'], df_media['clean']])
all_labels = np.array(['public'] * len(df_public) + ['media'] * len(df_media))

clf_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
X = clf_tfidf.fit_transform(all_texts)
y = all_labels

# Naive Bayes with GridSearchCV
nb_params = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}
nb_grid = GridSearchCV(MultinomialNB(), nb_params, cv=5, scoring='accuracy')
nb_grid.fit(X, y)
nb_best = nb_grid.best_estimator_
nb_score = nb_grid.best_score_
print(f"Naive Bayes best alpha: {nb_grid.best_params_['alpha']}, CV accuracy: {nb_score:.4f}")

# KNN with k selection
k_range = range(3, 21, 2)
knn_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5, scoring='accuracy').mean()
    knn_scores.append(score)

best_k_idx = np.argmax(knn_scores)
best_k = list(k_range)[best_k_idx]
knn_best_score = knn_scores[best_k_idx]
print(f"KNN best k: {best_k}, CV accuracy: {knn_best_score:.4f}")

# Figure: KNN k selection
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(list(k_range), knn_scores, 'o-', color='#2196F3', linewidth=2)
ax.axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
ax.set_xlabel('k (Number of Neighbors)')
ax.set_ylabel('Cross-Validation Accuracy')
ax.set_title('KNN: Accuracy vs k', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/knn_k_selection.png')
plt.close()
print("Saved knn_k_selection.png")

# Figure: Classification Comparison
fig, ax = plt.subplots(figsize=(8, 5))
models = ['Naive Bayes', f'KNN (k={best_k})']
scores = [nb_score, knn_best_score]
bars = ax.bar(models, scores, color=['#2196F3', '#FF9800'], alpha=0.8, width=0.5)
ax.set_ylabel('Cross-Validation Accuracy')
ax.set_title('Classification Performance: Public vs Media', fontweight='bold')
ax.set_ylim(0.5, 1.0)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{score:.3f}', ha='center', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/classification_comparison.png')
plt.close()
print("Saved classification_comparison.png")

# Top discriminative features
feature_names = clf_tfidf.get_feature_names_out()
nb_log_probs = nb_best.feature_log_prob_
media_idx = list(nb_best.classes_).index('media')
public_idx = list(nb_best.classes_).index('public')
log_ratio = nb_log_probs[public_idx] - nb_log_probs[media_idx]

top_public_features = [feature_names[i] for i in log_ratio.argsort()[-10:][::-1]]
top_media_features = [feature_names[i] for i in log_ratio.argsort()[:10]]
print(f"Top public-indicative features: {top_public_features}")
print(f"Top media-indicative features: {top_media_features}")


# =============================================================================
# NOTEBOOK 07: TOPIC MODELING
# =============================================================================
print("\n" + "=" * 60)
print("NOTEBOOK 07: TOPIC MODELING")
print("=" * 60)

# LDA Topic Modeling
lda_tfidf = TfidfVectorizer(max_features=3000, min_df=5)
pub_lda_matrix = lda_tfidf.fit_transform(df_public['clean'])

# Topic selection via perplexity
topic_range = range(3, 9)
perplexities = []
for n in topic_range:
    lda = LatentDirichletAllocation(n_components=n, random_state=42, max_iter=20)
    lda.fit(pub_lda_matrix)
    perplexities.append(lda.perplexity(pub_lda_matrix))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(topic_range), perplexities, 'o-', color='#2196F3', linewidth=2)
ax.set_xlabel('Number of Topics')
ax.set_ylabel('Perplexity')
ax.set_title('LDA: Perplexity vs Number of Topics', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lda_topic_selection.png')
plt.close()
print("Saved lda_topic_selection.png")

# Fit LDA with 5 topics for each corpus
n_topics = 5
lda_pub = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=30)
lda_pub.fit(pub_lda_matrix)

lda_tfidf_med = TfidfVectorizer(max_features=3000, min_df=3)
med_lda_matrix = lda_tfidf_med.fit_transform(df_media['clean'])
lda_med = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=30)
lda_med.fit(med_lda_matrix)

print("\nPublic Topics:")
pub_feature_names = lda_tfidf.get_feature_names_out()
for i, topic in enumerate(lda_pub.components_):
    top_words = [pub_feature_names[j] for j in topic.argsort()[-8:][::-1]]
    print(f"  Topic {i + 1}: {', '.join(top_words)}")

print("\nMedia Topics:")
med_feature_names = lda_tfidf_med.get_feature_names_out()
for i, topic in enumerate(lda_med.components_):
    top_words = [med_feature_names[j] for j in topic.argsort()[-8:][::-1]]
    print(f"  Topic {i + 1}: {', '.join(top_words)}")

# Figure: Topic Distributions
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, lda_model, feat_names, title, color in [
    (axes[0], lda_pub, pub_feature_names, 'Public Topics', '#2196F3'),
    (axes[1], lda_med, med_feature_names, 'Media Topics', '#FF9800')
]:
    for i, topic in enumerate(lda_model.components_):
        top_idx = topic.argsort()[-6:][::-1]
        words = [feat_names[j] for j in top_idx]
        weights = [topic[j] for j in top_idx]
        y_pos = np.arange(len(words)) + i * 7
        ax.barh(y_pos, weights, color=color, alpha=0.6 + 0.08 * i)
        for j, (w, p) in enumerate(zip(words, y_pos)):
            ax.text(0.01, p, w, va='center', fontsize=8)
    ax.set_title(title, fontweight='bold')
    ax.set_yticks([])
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/topic_distributions.png')
plt.close()
print("Saved topic_distributions.png")

# K-Means Clustering
combined_km_tfidf = TfidfVectorizer(max_features=3000, min_df=3)
combined_km_matrix = combined_km_tfidf.fit_transform(pd.concat([df_public['clean'], df_media['clean']]))

# Elbow method
inertias = []
k_range_km = range(2, 10)
for k in k_range_km:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
    km.fit(combined_km_matrix)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(k_range_km), inertias, 'o-', color='#2196F3', linewidth=2)
ax.set_xlabel('k (Number of Clusters)')
ax.set_ylabel('Inertia')
ax.set_title('K-Means: Elbow Method', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/kmeans_selection.png')
plt.close()
print("Saved kmeans_selection.png")

# Cluster purity with k=2
km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
km2_labels = km2.fit_predict(combined_km_matrix)
true_labels = np.array([0] * len(df_public) + [1] * len(df_media))

# Purity
cluster_0_public = (km2_labels[:len(df_public)] == 0).sum()
cluster_0_media = (km2_labels[len(df_public):] == 0).sum()
cluster_1_public = (km2_labels[:len(df_public)] == 1).sum()
cluster_1_media = (km2_labels[len(df_public):] == 1).sum()

purity = (max(cluster_0_public, cluster_0_media) + max(cluster_1_public, cluster_1_media)) / len(km2_labels)
print(f"\nK-Means (k=2) Purity: {purity:.3f}")


# =============================================================================
# SAVE ANALYSIS STATS
# =============================================================================
print("\n" + "=" * 60)
print("SAVING ANALYSIS STATS")
print("=" * 60)

analysis_stats = {
    "public_mean_sentiment": round(df_public['sentiment'].mean(), 4),
    "media_mean_sentiment": round(df_media['sentiment'].mean(), 4),
    "t_statistic": round(t_stat, 3),
    "p_value": round(p_val, 6),
    "cohens_d": round(cohens_d, 3),
    "cosine_similarity": round(cos_sim_score, 3),
    "public_n": len(df_public),
    "media_n": len(df_media),
    "reddit_n": len(df_reddit),
    "webmd_n": len(df_webmd),
    "news_n": len(df_news),
    "public_sentiment_dist": pub_dist,
    "media_sentiment_dist": med_dist,
    "nb_accuracy": round(nb_score, 4),
    "nb_best_alpha": nb_grid.best_params_['alpha'],
    "knn_accuracy": round(knn_best_score, 4),
    "knn_best_k": best_k,
    "kmeans_purity": round(purity, 3),
    "top_public_features": top_public_features,
    "top_media_features": top_media_features,
}

with open(f'{DATA_DIR}/analysis_stats.json', 'w') as f:
    json.dump(analysis_stats, f, indent=2)

print(json.dumps(analysis_stats, indent=2))
print("\nAll analysis complete. All figures regenerated.")
