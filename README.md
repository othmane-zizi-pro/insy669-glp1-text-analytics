# Media vs Public Opinion on GLP-1 Weight Loss Drugs (Ozempic & Wegovy)

## INSY 669 - Text Analytics | McGill University | Winter 2026

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Project Overview

This project applies text analytics techniques to compare **media coverage** versus **public opinion** surrounding GLP-1 receptor agonist weight loss drugs, specifically **Ozempic** (semaglutide) and **Wegovy** (semaglutide for weight management). 

GLP-1 drugs have experienced explosive growth in popularity, but how the media frames these medications may differ significantly from how actual users experience them. We analyze ~1,500 documents across two corpora to uncover these differences.

## Team Members

| Name | Student ID |
|------|-----------|
| Vasilis Christopoulos | 261278396 |
| Hugo Guideau | 261261108 |
| Saksi Khosla | 261284778 |
| Mustafa Yousuf | 261265412 |
| Othmane Zizi | 261255341 |

## Data Sources

### Public Opinion Corpus (1,100 documents)
- **Reddit** (800 posts): r/Ozempic, r/Semaglutide, r/WegovyWeightLoss
- **WebMD** (300 reviews): Patient reviews for Ozempic and Wegovy

### Media Corpus (400 documents)
- **News Articles**: Reuters, CNN Health, NYT, STAT News, Medical News Today, NPR Health, and others

## Methodology

1. **Data Collection**: Web scraping using BeautifulSoup, PRAW (Reddit API), and NewsAPI
2. **Text Preprocessing**: Tokenization, stopword removal, lemmatization (NLTK)
3. **Feature Extraction**: Bag-of-Words, TF-IDF representations (unigrams + bigrams)
4. **Sentiment Analysis**: VADER sentiment scoring with statistical testing (t-test, Cohen's d)
5. **Word Associations**: PMI (Pointwise Mutual Information) and Lift metrics
6. **Corpus Comparison**: TF-IDF keyword analysis, cosine similarity, MDS visualization, side effects gap analysis
7. **Text Classification**: Naive Bayes (Multinomial) and K-Nearest Neighbors classifiers to distinguish media from public text, with hyperparameter tuning via GridSearchCV and cross-validation
8. **Topic Modeling & Clustering**: LDA (Latent Dirichlet Allocation) for latent theme discovery in each corpus, K-Means clustering on combined TF-IDF to evaluate natural corpus separation

## Repository Structure

```
├── README.md
├── requirements.txt
├── proposal/
│   └── group-project-proposal.pdf
├── data/
│   ├── reddit_posts.csv
│   ├── webmd_reviews.csv
│   ├── news_articles.csv
│   ├── public_processed.csv
│   ├── media_processed.csv
│   └── analysis_stats.json
├── notebooks/
│   ├── 01-data-collection.ipynb      # Scraping Reddit, WebMD, news APIs
│   ├── 02-preprocessing.ipynb        # Text cleaning, tokenization, vectorization
│   ├── 03-sentiment.ipynb            # VADER sentiment analysis + statistical tests
│   ├── 04-associations.ipynb         # PMI and Lift word associations
│   ├── 05-comparison.ipynb           # TF-IDF comparison, word clouds, side effects
│   ├── 06-classification.ipynb       # Naive Bayes + KNN text classification
│   └── 07-topic-modeling.ipynb       # LDA topic modeling + K-Means clustering
├── figures/
│   ├── sentiment_histograms.png
│   ├── sentiment_boxplot.png
│   ├── sentiment_pies.png
│   ├── tfidf_comparison.png
│   ├── wordclouds.png
│   ├── mds_plot.png
│   ├── side_effects.png
│   ├── sentiment_timeline.png
│   ├── knn_k_selection.png
│   ├── classification_comparison.png
│   ├── lda_topic_selection.png
│   ├── topic_distributions.png
│   └── kmeans_selection.png
└── presentation/
    ├── presentation.html
    ├── presentation.pdf
    └── presentation.pptx
```

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/othmane-zizi-pro/insy669-glp1-text-analytics.git
cd insy669-glp1-text-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# 4. Run notebooks in order
jupyter notebook notebooks/
```

## Key Findings

1. **Sentiment Gap**: Public opinion shows more polarized sentiment (both highly positive and highly negative) compared to media's more neutral, measured tone.

2. **Language Differences**: Public discourse focuses on personal experiences (weight loss numbers, side effects, costs), while media emphasizes clinical trials, market dynamics, and regulatory issues.

3. **Side Effects**: Users frequently discuss nausea, constipation, and sulfur burps - side effects that receive less proportional coverage in media articles.

4. **Cost Concerns**: Affordability and insurance coverage are dominant themes in public discussion but treated as secondary topics in media coverage.

5. **Classifiability**: Naive Bayes and KNN classifiers can distinguish media from public text with high accuracy, confirming that the two corpora use fundamentally different language patterns.

6. **Latent Topics**: LDA reveals distinct thematic structures in each corpus - public topics center on personal health journeys while media topics revolve around industry and clinical narratives. K-Means clustering naturally separates the two corpora.

## License

This project was created for academic purposes as part of INSY 669 at McGill University.
