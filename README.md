# Media vs. Public Opinion on GLP-1 Weight-Loss Drugs (Ozempic & Wegovy)

**INSY 669 — Text Analytics | McGill University | Winter 2026**

| Name | Student ID |
|------|-----------|
| Vasilis Christopoulos | 261278396 |
| Hugo Guideau | 261261108 |
| Saksi Khosla | 261284778 |
| Mustafa Yousuf | 261265412 |
| Othmane Zizi | 261255341 |

---

## What This Project Does

Compares **public discourse** (Reddit posts + WebMD patient reviews) with **media discourse** (Google News articles) about GLP-1 weight-loss drugs during Jan–Nov 2024, using an 8-stage NLP/ML pipeline: sentiment analysis, word associations, TF-IDF comparison, supervised classification, topic modelling, aspect-based sentiment, side-effect gap analysis, and temporal lead-lag testing.

---

## Quick Start — Reproduce Everything in 4 Commands

> **Requirements:** Python 3.9+ and an internet connection (for NLTK downloads).
> **Expected runtime:** ~1 minute.
> **All data is included** — no API keys or external downloads needed.

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download required NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# 3. Validate that everything is in place
python project_cli.py validate --media-text-mode hybrid

# 4. Run the full analysis pipeline
python project_cli.py run-analysis --media-text-mode hybrid
```

That's it. When step 4 finishes you will see updated outputs in `data/`, `figures/`, and a summary printed to the terminal.

### What Gets Generated

| Output | Location | Description |
|--------|----------|-------------|
| 27 figures | `figures/*.png` | Sentiment plots, TF-IDF charts, classification results, topic distributions, side-effect comparisons, temporal analysis |
| Processed data | `data/public_processed.csv`, `data/media_processed.csv` | Tokenised and cleaned corpora |
| Sentiment data | `data/public_with_sentiment.csv`, `data/media_with_sentiment.csv` | Documents with VADER scores |
| Statistics JSON | `data/analysis_stats.json` | Every metric referenced in the report (sentiment means, p-values, accuracies, etc.) |
| Report | `report/report.pdf`, `report/report.docx` | 2-page summary with figures and tables |
| Report v2 | `report/report_v2.pdf`, `report/report_v2.tex` | Fresh-hybrid update based on the latest validated recollection snapshot |

---

## Repository Structure

```
├── project_cli.py            ← CLI entrypoint (start here)
├── run_all_analysis.py       ← 8-stage analysis pipeline
├── collect_real_data.py      ← Data collection (Reddit, WebMD, News)
├── analysis_utils.py         ← Shared utilities (date parsing, schema validation)
├── requirements.txt          ← Python dependencies
├── README.md
│
├── data/                     ← Raw + processed datasets (all included)
│   ├── reddit_posts.csv          3,246 Reddit posts
│   ├── webmd_reviews.csv         1,000 WebMD reviews (102 after 2024 filter)
│   ├── news_articles.csv         634 Google News articles
│   ├── public_processed.csv      Preprocessed public corpus
│   ├── media_processed.csv       Preprocessed media corpus
│   ├── *_with_sentiment.csv      Corpora with VADER scores
│   └── analysis_stats.json       All computed metrics
│
├── figures/                  ← All 27 generated visualisations
│
├── notebooks/                ← Jupyter notebooks for exploration
│   ├── 01-data-collection.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-sentiment.ipynb
│   ├── 04-associations.ipynb
│   ├── 05-comparison.ipynb
│   ├── 06-classification.ipynb
│   ├── 07-topic-modeling.ipynb
│   ├── 08-aspect-sentiment.ipynb
│   └── 09-temporal-leadlag.ipynb
│
└── report/                   ← Final 2-page report
    ├── report.tex                LaTeX source
    ├── report.pdf                Compiled PDF
    ├── report.docx               Word version
    ├── report_v2.tex             LaTeX source (fresh hybrid update)
    ├── report_v2.pdf             Compiled PDF (fresh hybrid update)
    └── generate_docx.py          Script that builds the DOCX
```

---

## Pipeline Details

### Data Sources

| Source | Corpus | n | Period | Collection |
|--------|--------|---|--------|------------|
| Reddit (r/Ozempic, r/Semaglutide, r/WegovyWeightLoss) | Public | 3,246 | Jan–Nov 2024 | Arctic Shift API |
| WebMD patient reviews | Public | 102 (after 2024 filter) | 2024 | Web scraping |
| Google News RSS | Media | 634 | Jan–Nov 2024 | RSS + URL decoding |

### Analysis Stages

The script `run_all_analysis.py` executes these stages sequentially (each corresponds to a notebook):

| # | Stage | What It Does | Key Outputs |
|---|-------|-------------|-------------|
| 1 | Preprocessing | Tokenise, lowercase, remove stopwords, lemmatise | `clean` and `clean_norm40` columns |
| 2 | Sentiment | VADER compound scores, t-test, Cohen's d, Mann-Whitney U | `sentiment_histograms.png`, `sentiment_boxplot.png` |
| 3 | Associations | PMI for key terms, MDS document similarity | `pmi_*.png`, `mds_plot.png` |
| 4 | Comparison | TF-IDF top terms, word clouds, cosine similarity, side-effect gap | `tfidf_comparison.png`, `wordclouds.png`, `side_effects.png` |
| 5 | Classification | Naive Bayes + KNN with `StratifiedKFold` CV (leakage-safe) | `classification_comparison.png`, `knn_k_selection.png` |
| 6 | Topic Modelling | LDA (5 topics/corpus), K-Means clustering | `topic_distributions.png`, `kmeans_selection.png` |
| 7 | Aspect Sentiment | Sentiment across 7 health aspects, logistic regression | `aspect_sentiment_comparison.png`, `aspect_discrepancy.png` |
| 8 | Temporal | Granger causality, cross-correlation, volume trends | `granger_results.png`, `crosscorrelation.png` |

### Hybrid Media-Text Mode

News RSS feeds only provide headlines + short snippets, which can bias analysis. The `--media-text-mode` flag controls how media text is selected:

- **`snippet`** — title + description only (baseline)
- **`body`** — full article body only (requires pre-extraction via `--fetch-news-body`)
- **`hybrid`** (default, recommended) — uses full body when available (≥80 tokens), falls back to snippet

### Length-Normalised Robustness

Every metric is computed on both full text and a 40-token-capped version (`clean_norm40`). If results are consistent across both, they are not driven by document-length differences. Key normalised outputs:
- `figures/tfidf_comparison_normalized.png`
- `figures/classification_comparison_normalized.png`
- `figures/side_effects_normalized_rate.png`

---

## Key Findings

| Finding | Evidence |
|---------|----------|
| **Sentiment gap** | Public +0.121 vs. media −0.137 (t = 9.43, p < 0.001, Cohen's d = 0.44) |
| **High classifiability** | Naive Bayes 97.6%, KNN 96.7% — stable under length normalisation |
| **Side-effect coverage gap** | Nausea, constipation, anxiety discussed 4–10x more by patients than media |
| **Aspect discrepancy** | Largest gaps in mental health (0.63) and access (0.55) |
| **No temporal causation** | Granger tests non-significant in both directions |
| **Natural clustering** | K-Means (k=2) purity 84.1% without supervision |

---

## Notebooks vs. Scripts

The **9 notebooks** are for exploration and step-by-step walkthrough — each one maps to a pipeline stage. The **script pipeline** (`project_cli.py` → `run_all_analysis.py`) is the canonical reproducibility path and generates all final outputs. Both use the same logic; the script is authoritative.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `validate` fails on NLTK resources | Run the NLTK download command from step 2 |
| `ModuleNotFoundError` | Make sure the virtual environment is activated (`source .venv/bin/activate`) |
| `python` not found | Use `python3` instead |
| Want to recollect data from scratch | `python project_cli.py run-all --fetch-news-body --media-text-mode hybrid` (requires internet; may take several minutes) |
| `media_text_mode=body` shows 0 documents | Recollect with `--fetch-news-body` first so `news_articles.csv` includes `text_body` |

---

*Created for INSY 669 — Text Analytics at McGill University, Winter 2026.*
