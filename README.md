# Media vs Public Opinion on GLP-1 Weight Loss Drugs (Ozempic & Wegovy)

## INSY 669 - Text Analytics | McGill University | Winter 2026

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Project Overview
This project compares language and sentiment in:
- **Public discourse**: Reddit + WebMD reviews
- **Media discourse**: Google News RSS article snippets with optional full-article body enrichment

The repository includes:
- data collection scripts
- notebook workflow for exploration
- a reproducible script-first analysis pipeline that regenerates outputs end-to-end

## What Changed (Methodology Fixes)
This repository now includes the following major methodology improvements:

1. **Schema/date reliability across script + notebooks**
- Shared utilities in `analysis_utils.py` standardize date parsing and sentiment schema.
- Pipeline enforces required columns and surfaces clear errors when inputs are malformed.

2. **Leakage-safe classification**
- Naive Bayes and KNN now run in `Pipeline` objects with `StratifiedKFold` + `GridSearchCV`.
- No global pre-fit vectorizer is reused across folds.

3. **Media text confound mitigation**
- News collection supports optional full-body extraction (`text_body`) with decoded publisher URLs.
- Analysis supports `--media-text-mode {snippet,body,hybrid}`.
- `hybrid` is recommended: use `text_body` when available, fallback to snippet.

4. **Length-normalized robustness track**
- A normalized analysis track (`clean_norm40` by default) remains available for confound checks.
- Normalized metrics are saved alongside full-text metrics.

5. **Script-first UX**
- `project_cli.py` is the canonical entrypoint for validation, analysis-only, and recollect+analysis runs.

## Canonical Workflow (Recommended)
The canonical path is **script-first** using `project_cli.py`.

### 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Validate environment and files
```bash
python project_cli.py validate
```

### 3. Run analysis from existing data
```bash
python project_cli.py run-analysis --media-text-mode hybrid
```

### 4. (Optional) Recollect data then run analysis
```bash
python project_cli.py run-all --fetch-news-body --media-text-mode hybrid
```

`media-text-mode` options:
- `snippet`: use title + description only
- `body`: use `text_body` only (requires body-enriched news data)
- `hybrid` (recommended): use `text_body` when available, otherwise fallback to snippet

## Sensitivity Workflow (Recommended for Reporting)
Run all three media modes on the **same frozen dataset** and compare outputs:

```bash
mkdir -p results

python project_cli.py run-analysis --media-text-mode snippet --media-body-min-tokens 80 --normalized-token-cap 40
cp data/analysis_stats.json results/analysis_stats_snippet.json

python project_cli.py run-analysis --media-text-mode hybrid --media-body-min-tokens 80 --normalized-token-cap 40
cp data/analysis_stats.json results/analysis_stats_hybrid.json

python project_cli.py run-analysis --media-text-mode body --media-body-min-tokens 80 --normalized-token-cap 40
cp data/analysis_stats.json results/analysis_stats_body.json
```

Reporting rule of thumb:
- Use `hybrid` as primary.
- Treat findings as strong only if direction is stable across `snippet/hybrid/body` and full vs normalized tracks.

## Notebook Workflow (Secondary / Exploratory)
Notebooks are kept for exploration and presentation.  
Primary reproducibility should come from the script workflow above.

Notebook order:
1. `notebooks/01-data-collection.ipynb`
2. `notebooks/02-preprocessing.ipynb`
3. `notebooks/03-sentiment.ipynb`
4. `notebooks/04-associations.ipynb`
5. `notebooks/05-comparison.ipynb`
6. `notebooks/06-classification.ipynb`
7. `notebooks/07-topic-modeling.ipynb`
8. `notebooks/08-aspect-sentiment.ipynb`
9. `notebooks/09-temporal-leadlag.ipynb`

## Data Sources
- **Reddit**: Arctic Shift API (subreddits around Ozempic/Wegovy/Semaglutide)
- **WebMD**: scraped patient reviews
- **Media**: Google News RSS search results
  - Base schema keeps snippet text for compatibility (`text`)
  - Optional enrichment adds `text_body` (publisher article text), `article_url`, and `text_snippet`

## Method Summary
1. Data collection and schema normalization (with optional media full-body enrichment)
2. Text preprocessing (tokenization, stopword removal, lemmatization)
3. Sentiment analysis (VADER)
4. Associations/comparison (PMI, TF-IDF, cosine similarity, side-effect coverage)
5. Classification (Naive Bayes + KNN with leakage-safe CV pipelines)
6. Topic modeling/clustering (LDA, K-Means)
7. Aspect-based sentiment
8. Temporal lead-lag analysis
9. **Length-normalized robustness track** (`clean_norm40`) to reduce format/length confounding
10. **Media text mode robustness** (`snippet` vs `body` vs `hybrid`) to reduce headline-only bias

## Key Outputs
- Processed datasets: `data/public_processed.csv`, `data/media_processed.csv`
- Sentiment datasets: `data/public_with_sentiment.csv`, `data/media_with_sentiment.csv`
- Stats summary: `data/analysis_stats.json`
- Figures: `figures/*.png` (baseline + normalized robustness figures)

Notable normalized outputs:
- `figures/tfidf_comparison_normalized.png`
- `figures/classification_comparison_normalized.png`
- `figures/side_effects_normalized_rate.png`
- JSON keys:
  - `cosine_similarity_normalized`
  - `nb_accuracy_normalized`
  - `knn_accuracy_normalized`
  - `side_effect_rates_per_1k_tokens`
  - `side_effect_doc_prevalence`
  - `media_text_mode`
  - `media_body_coverage`
  - `media_docs_fallback_to_snippet`

## Troubleshooting
- **`validate` fails on NLTK resources**: run the NLTK download command in setup.
- **Missing `data/*.csv` for `run-analysis`**: run `python project_cli.py run-all` or place expected files in `data/`.
- **`media_text_mode=body` fails**: recollect with `python project_cli.py run-all --fetch-news-body` so `data/news_articles.csv` includes `text_body`.
- **`run-all --fetch-news-body` is slow**: full-body extraction depends on publisher response times and can take several minutes.
- **Interrupted collection leaves partial files**: rerun the specific collector (for example `python collect_webmd_real.py`) before analysis.
- **Date parsing issues in notebooks**: use updated notebooks that import `analysis_utils.py`.
- **Long runtime**: classification uses grid-search cross-validation and can take several minutes.

## Repository Structure
```text
├── project_cli.py
├── run_all_analysis.py
├── analysis_utils.py
├── collect_real_data.py
├── collect_reddit_v2.py
├── collect_webmd_real.py
├── collect_webmd_and_clean.py
├── requirements.txt
├── data/
├── figures/
├── notebooks/
├── proposal/
│   └── group-project-proposal.pdf
└── presentation/
    └── presentation.pptx
```

## License
Created for academic purposes as part of INSY 669 at McGill University.
