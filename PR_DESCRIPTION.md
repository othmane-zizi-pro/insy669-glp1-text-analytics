# PR Title
Methodology Hardening + Media Full-Text Enrichment + Script-First Reproducibility

## Why This PR
This PR addresses critical methodology and reproducibility issues in the GLP-1 text analytics pipeline:

1. Fragile date/schema handling between scripts and notebooks.
2. Classification leakage from pre-fitting vectorizers before CV.
3. Format/length confound between short media snippets and longer public posts.
4. Ambiguous user workflow between notebooks and scripts.

It also introduces richer media-text handling (`snippet|body|hybrid`) so comparisons are less headline-driven.

## Scope
In scope:
- Schema/date standardization and validation.
- Leakage-safe classification evaluation.
- Length-normalized robustness track.
- Optional full-body media enrichment and selectable media analysis text mode.
- Script-first CLI workflow and updated docs.

Out of scope:
- Redesigning source sampling composition.
- Replacing datasets with a new sampling strategy.

## What Was Implemented

### 1) Schema + Date Reliability
- Added shared utility module: `analysis_utils.py`.
- Standardized date parsing and ISO date output.
- Enforced sentiment schema compatibility:
  - `sentiment` numeric
  - `compound` alias
  - `sentiment_label`
- Added explicit input column validation with clear errors.

Files:
- `analysis_utils.py`
- `run_all_analysis.py`
- `notebooks/03-sentiment.ipynb`
- `notebooks/05-comparison.ipynb`
- `notebooks/08-aspect-sentiment.ipynb`
- `notebooks/09-temporal-leadlag.ipynb`

### 2) Classification Leakage Removal
- Replaced leakage-prone flows with CV-safe `Pipeline` patterns.
- Used `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Added `GridSearchCV` tuning for NB (`alpha`) and KNN (`n_neighbors`).

Files:
- `run_all_analysis.py`
- `notebooks/06-classification.ipynb`

### 3) Length-Confound Robustness Track
- Added normalized text track (`clean_norm40`) for public and media.
- Added normalized metrics and figures:
  - `figures/tfidf_comparison_normalized.png`
  - `figures/classification_comparison_normalized.png`
  - `figures/side_effects_normalized_rate.png`
- Extended `analysis_stats.json` with normalized keys:
  - `cosine_similarity_normalized`
  - `nb_accuracy_normalized`
  - `knn_accuracy_normalized`
  - `side_effect_rates_per_1k_tokens`
  - `side_effect_doc_prevalence`

Files:
- `run_all_analysis.py`
- `notebooks/05-comparison.ipynb`

### 4) Media Full-Text Enrichment + Mode Selection
- Added optional Google News URL decoding + article body extraction.
- `news_articles.csv` can now include:
  - `text` (legacy snippet)
  - `text_snippet`
  - `text_body`
  - `body_token_count`
  - `rss_link`, `article_url`, `title`, `description`
- Added analysis mode switch:
  - `--media-text-mode snippet|body|hybrid`
  - `--media-body-min-tokens` threshold for usable body text.
- `hybrid` mode uses body when available, otherwise snippet fallback.

Files:
- `collect_real_data.py`
- `collect_webmd_and_clean.py`
- `run_all_analysis.py`
- `requirements.txt` (added `googlenewsdecoder`)

### 5) Script-First UX + Validation
- Added canonical CLI entrypoint `project_cli.py`:
  - `validate`
  - `run-analysis`
  - `run-all`
- Validation now checks media schema compatibility for requested mode.
- CLI passes mode and normalization flags through to analysis.

Files:
- `project_cli.py`
- `README.md`

## How This Fixes the Critical Issues

1. **Schema/date breaks** are prevented by central parsing/validation.
2. **Classification leakage** is removed by fold-isolated vectorization in `Pipeline`.
3. **Format confound** is reduced via:
   - richer media text (`body`/`hybrid`)
   - normalized robustness track.
4. **Run confusion** is resolved by single CLI-driven canonical workflow.

## Behavior Changes

1. Analysis now records mode metadata in `analysis_stats.json`:
- `media_text_mode`
- `media_body_min_tokens`
- `media_docs_with_usable_body`
- `media_body_coverage`
- `media_docs_after_mode_filter`
- `media_docs_fallback_to_snippet`
- `normalized_token_cap`

2. `run-analysis` can fail early if `body` mode is requested without `text_body`.

3. `run-all --fetch-news-body` can take significantly longer due publisher fetch latency.

## Backward Compatibility

Maintained:
- Existing output files still generated.
- Existing sentiment columns remain (`sentiment` retained; `compound` guaranteed).
- Legacy `news_articles.csv:text` semantics preserved as snippet text.

Additive:
- Extra columns/metadata are added, not replacing legacy contract.

## Testing Performed

Command-level checks:
- `python -m py_compile run_all_analysis.py collect_real_data.py collect_webmd_and_clean.py project_cli.py analysis_utils.py`
- `python project_cli.py --help`
- `python project_cli.py validate --media-text-mode hybrid`
- `python project_cli.py validate --media-text-mode body` (expected fail without `text_body`)
- `python project_cli.py run-analysis --media-text-mode hybrid`
- `python project_cli.py run-analysis --media-text-mode body` (after enrichment)

Richer media validation:
- Confirmed `news_articles.csv` contained `text_body`.
- Confirmed coverage and mode metadata written to `analysis_stats.json`.
- Confirmed end-to-end completion in both `hybrid` and `body` modes.

## Known Limitations

1. Body extraction quality varies by publisher (paywalls/boilerplate/403).
2. Temporal causality outputs are sensitive to representation mode and should be treated as exploratory.
3. If collection is interrupted, partial CSVs can appear and should be regenerated.

## Suggested Next Steps

1. Add a small evaluation sample for body-extraction quality (precision/recall style audit).
2. Add a one-command sensitivity runner that exports `snippet`, `hybrid`, `body` stats side-by-side.
3. Freeze dataset snapshots for report reproducibility (timestamped `data_snapshot` folder).
4. Add deduplication for syndicated news bodies (near-duplicate detection).
5. Consider replacing hard `norm40` with a configurable sweep (40/60/80/120/full).

## Reviewer Checklist

1. Verify mode handling (`snippet|body|hybrid`) in `run_all_analysis.py`.
2. Verify leakage-safe CV setup in classification.
3. Verify `analysis_stats.json` includes new mode + normalized keys.
4. Verify README commands run as documented.
5. Verify backward compatibility for downstream consumers expecting `sentiment` and `text`.
