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

Compares **public discourse** (Reddit posts + WebMD patient reviews) with **media discourse** (Google News articles) about GLP-1 weight-loss drugs during Jan–Nov 2024, using a 9-stage NLP/ML pipeline: sentiment analysis, word associations, TF-IDF comparison, supervised classification, **public subanalysis (Reddit vs WebMD)**, topic modelling, aspect-based sentiment, side-effect gap analysis, and temporal lead-lag testing.

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
| 30 figures | `figures/*.png` | Sentiment plots, TF-IDF charts, classification results, topic distributions, side-effect comparisons, temporal analysis, and Reddit-vs-WebMD subanalysis visuals |
| Processed data | `data/public_processed.csv`, `data/media_processed.csv` | Tokenised and cleaned corpora |
| Sentiment data | `data/public_with_sentiment.csv`, `data/media_with_sentiment.csv` | Documents with VADER scores |
| Statistics JSON | `data/analysis_stats.json` | Every metric referenced in the report (sentiment means, p-values, accuracies, etc.) |
| Report | `report/report.pdf`, `report/report.docx` | 2-page summary with figures and tables |
| Report v2 | `report/report_v2.pdf`, `report/report_v2.tex` | Fresh-hybrid update based on the latest validated recollection snapshot |
| Report v3 | `report/report_v3.pdf`, `report/report_v3.tex` | Adds Reddit-vs-WebMD public subanalysis while preserving full pipeline coverage |

---

## Repository Structure

```
├── project_cli.py            ← CLI entrypoint (start here)
├── run_all_analysis.py       ← 9-stage analysis pipeline
├── collect_real_data.py      ← Data collection (Reddit, WebMD, News)
├── analysis_utils.py         ← Shared utilities (date parsing, schema validation)
├── requirements.txt          ← Python dependencies
├── README.md
│
├── data/                     ← Raw + processed datasets (all included)
│   ├── reddit_posts.csv          1,184 Reddit posts
│   ├── webmd_reviews.csv         102 WebMD reviews
│   ├── news_articles.csv         647 Google News articles
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
    ├── report_v3.tex             LaTeX source (public subanalysis update)
    ├── report_v3.pdf             Compiled PDF (public subanalysis update)
    └── generate_docx.py          Script that builds the DOCX
```

---

## Pipeline Details

### Data Sources

| Source | Corpus | n | Period | Collection |
|--------|--------|---|--------|------------|
| Reddit (r/Ozempic, r/Semaglutide, r/WegovyWeightLoss) | Public | 1,184 | Jan–Nov 2024 | Arctic Shift API |
| WebMD patient reviews | Public | 102 (after 2024 filter) | 2024 | Web scraping |
| Google News RSS | Media | 647 | Jan–Nov 2024 | RSS + URL decoding |

### Analysis Stages

The script `run_all_analysis.py` executes these stages sequentially (each corresponds to a notebook):

| # | Stage | What It Does | Key Outputs |
|---|-------|-------------|-------------|
| 1 | Preprocessing | Tokenise, lowercase, remove stopwords, lemmatise | `clean` and `clean_norm40` columns |
| 2 | Sentiment | VADER compound scores, t-test, Cohen's d, Mann-Whitney U | `sentiment_histograms.png`, `sentiment_boxplot.png` |
| 3 | Associations | PMI for key terms, MDS document similarity | `pmi_*.png`, `mds_plot.png` |
| 4 | Comparison | TF-IDF top terms, word clouds, cosine similarity, side-effect gap | `tfidf_comparison.png`, `wordclouds.png`, `side_effects.png` |
| 5 | Classification | Naive Bayes + KNN with `StratifiedKFold` CV (leakage-safe) | `classification_comparison.png`, `knn_k_selection.png` |
| 6 | Public Subanalysis | Reddit-vs-WebMD sentiment, lexical, and classification contrast | `reddit_webmd_sentiment_boxplot.png`, `reddit_webmd_tfidf_comparison.png`, `reddit_webmd_classification.png` |
| 7 | Topic Modelling | LDA (5 topics/corpus), K-Means clustering | `topic_distributions.png`, `kmeans_selection.png` |
| 8 | Aspect Sentiment | Sentiment across 7 health aspects, logistic regression | `aspect_sentiment_comparison.png`, `aspect_discrepancy.png` |
| 9 | Temporal | Granger causality, cross-correlation, volume trends | `granger_results.png`, `crosscorrelation.png` |

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
| **Sentiment contrast** | Public +0.1117 vs. media +0.1905 (t = -2.301, p = 0.0215, Cohen's d = -0.108) |
| **High classifiability** | Naive Bayes 97.83%, KNN 95.50% (normalized: 97.57% / 97.10%) |
| **Side-effect coverage gap** | Public mention prevalence exceeds media for nausea (8.4x), constipation (17.6x), and anxiety (3.0x) |
| **Aspect discrepancy** | Strongest public-minus-media gaps: cost (-0.187), dosage (-0.175), side effects (-0.174) |
| **No temporal causation** | Granger best p-values: media->public 0.7672, public->media 0.5183 (both non-significant) |
| **Natural clustering** | K-Means (k=2) purity 89.8% without supervision |
| **Within-public divergence** | Reddit vs WebMD sentiment differs (0.1479 vs -0.3077, p < 0.001), with high source separability (NB 92.22%, KNN 92.38%) |

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
