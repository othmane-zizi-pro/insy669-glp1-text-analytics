#!/usr/bin/env python3
"""Project CLI for validation and reproducible pipeline execution."""

from __future__ import annotations

import argparse
import csv
import importlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

ANALYSIS_INPUTS = [
    DATA_DIR / "reddit_posts.csv",
    DATA_DIR / "webmd_reviews.csv",
    DATA_DIR / "news_articles.csv",
]

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "sklearn",
    "nltk",
    "wordcloud",
    "bs4",
    "requests",
    "lxml",
    "statsmodels",
    "googlenewsdecoder",
]

NLTK_RESOURCES = {
    "punkt": ["tokenizers/punkt", "tokenizers/punkt.zip"],
    "punkt_tab": ["tokenizers/punkt_tab", "tokenizers/punkt_tab.zip"],
    "stopwords": ["corpora/stopwords", "corpora/stopwords.zip"],
    "wordnet": ["corpora/wordnet", "corpora/wordnet.zip", "corpora/wordnet.zip/wordnet/"],
    "vader_lexicon": [
        "sentiment/vader_lexicon",
        "sentiment/vader_lexicon.zip",
        "sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt",
    ],
}

MEDIA_TEXT_MODES = ("snippet", "body", "hybrid")


def _check_imports() -> tuple[list[str], list[str]]:
    ok = []
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            ok.append(pkg)
        except Exception:
            missing.append(pkg)
    return ok, missing


def _check_nltk_resources() -> tuple[list[str], list[str]]:
    try:
        import nltk
    except Exception:
        return [], list(NLTK_RESOURCES.keys())

    ok = []
    missing = []
    for name, locators in NLTK_RESOURCES.items():
        found = False
        for locator in locators:
            try:
                nltk.data.find(locator)
                found = True
                break
            except LookupError:
                continue
        if found:
            ok.append(name)
        else:
            missing.append(name)
    return ok, missing


def _check_writable_dir(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".cli_write_probe"
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            return next(reader, [])
    except Exception:
        return []


def _build_analysis_args(media_text_mode: str, media_body_min_tokens: int, normalized_token_cap: int) -> list[str]:
    return [
        "--media-text-mode",
        media_text_mode,
        "--media-body-min-tokens",
        str(max(1, media_body_min_tokens)),
        "--normalized-token-cap",
        str(max(1, normalized_token_cap)),
    ]


def validate(
    target: str,
    media_text_mode: str = "hybrid",
    fetch_news_body: bool = False,
) -> int:
    print("Running environment validation...")
    print(f"Project root: {ROOT}")
    print(f"Target: {target}")
    print(f"Media text mode: {media_text_mode}")

    _, missing_pkgs = _check_imports()
    _, missing_nltk = _check_nltk_resources()

    missing_inputs = []
    if target == "run-analysis":
        missing_inputs = [str(p) for p in ANALYSIS_INPUTS if not p.exists()]

    writable_data = _check_writable_dir(DATA_DIR)
    writable_fig = _check_writable_dir(FIG_DIR)

    if missing_pkgs:
        print(f"[FAIL] Missing Python packages: {missing_pkgs}")
    else:
        print("[OK] Python package imports")

    if missing_nltk:
        print(f"[FAIL] Missing NLTK resources: {missing_nltk}")
        print(
            "       Install with:\n"
            "       python -c \"import nltk; "
            "nltk.download('vader_lexicon'); nltk.download('punkt'); "
            "nltk.download('punkt_tab'); nltk.download('stopwords'); "
            "nltk.download('wordnet')\""
        )
    else:
        print("[OK] NLTK resources")

    if missing_inputs:
        print(f"[FAIL] Missing input files for analysis: {missing_inputs}")
    elif target == "run-analysis":
        print("[OK] Analysis input files")

    schema_failed = False
    if target == "run-analysis" and not missing_inputs:
        news_path = DATA_DIR / "news_articles.csv"
        news_cols = _read_csv_header(news_path)
        if not news_cols:
            print("[FAIL] Could not read header from data/news_articles.csv")
            schema_failed = True
        else:
            has_snippet = ("text" in news_cols) or ("text_snippet" in news_cols)
            has_body = "text_body" in news_cols
            if not has_snippet:
                print("[FAIL] news_articles.csv must include `text` or `text_snippet`.")
                schema_failed = True
            elif media_text_mode == "body" and not has_body:
                print(
                    "[FAIL] media_text_mode=body requires `text_body` in news_articles.csv.\n"
                    "       Recollect with: python project_cli.py run-all --fetch-news-body"
                )
                schema_failed = True
            elif media_text_mode == "hybrid" and not has_body:
                print("[WARN] `text_body` missing; hybrid mode will fall back to snippet text.")
            else:
                print("[OK] News schema supports requested media text mode")

    if writable_data and writable_fig:
        print("[OK] Output directories are writable")
    else:
        print(
            f"[FAIL] Output directories not writable: "
            f"data={writable_data}, figures={writable_fig}"
        )

    if fetch_news_body and "googlenewsdecoder" in missing_pkgs:
        print(
            "[FAIL] --fetch-news-body requires googlenewsdecoder. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    failed = bool(
        missing_pkgs
        or missing_nltk
        or missing_inputs
        or schema_failed
        or not (writable_data and writable_fig)
    )
    if failed:
        print("Validation failed.")
        return 1

    print("Validation passed.")
    return 0


def _run_script(script_name: str, script_args: list[str] | None = None) -> int:
    script_path = ROOT / script_name
    if not script_path.exists():
        print(f"[FAIL] Missing script: {script_path}")
        return 1
    cmd = [sys.executable, str(script_path)]
    if script_args:
        cmd.extend(script_args)
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def run_analysis(media_text_mode: str, media_body_min_tokens: int, normalized_token_cap: int) -> int:
    code = validate(target="run-analysis", media_text_mode=media_text_mode)
    if code != 0:
        return code
    return _run_script(
        "run_all_analysis.py",
        _build_analysis_args(media_text_mode, media_body_min_tokens, normalized_token_cap),
    )


def run_all(
    media_text_mode: str,
    media_body_min_tokens: int,
    normalized_token_cap: int,
    fetch_news_body: bool,
    news_body_timeout: int,
) -> int:
    code = validate(
        target="run-all",
        media_text_mode=media_text_mode,
        fetch_news_body=fetch_news_body,
    )
    if code != 0:
        return code

    collect_args: list[str] = []
    if fetch_news_body:
        collect_args.extend(
            [
                "--fetch-news-body",
                "--news-body-timeout",
                str(max(5, news_body_timeout)),
                "--min-body-tokens",
                str(max(10, media_body_min_tokens)),
            ]
        )

    code = _run_script("collect_real_data.py", collect_args)
    if code != 0:
        return code
    return _run_script(
        "run_all_analysis.py",
        _build_analysis_args(media_text_mode, media_body_min_tokens, normalized_token_cap),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for GLP-1 text analytics pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Validate env, resources, and files.")
    p_validate.add_argument(
        "--target",
        choices=["run-analysis", "run-all"],
        default="run-analysis",
        help="Validation profile to run.",
    )
    p_validate.add_argument(
        "--media-text-mode",
        choices=MEDIA_TEXT_MODES,
        default="hybrid",
        help="Validate compatibility with requested media text mode.",
    )
    p_validate.add_argument(
        "--fetch-news-body",
        action="store_true",
        help="Validate dependencies for full-body news collection.",
    )

    p_analysis = sub.add_parser("run-analysis", help="Run analysis pipeline from existing data files.")
    p_analysis.add_argument(
        "--media-text-mode",
        choices=MEDIA_TEXT_MODES,
        default="hybrid",
        help="Media text mode used by run_all_analysis.py.",
    )
    p_analysis.add_argument(
        "--media-body-min-tokens",
        type=int,
        default=80,
        help="Minimum token threshold for usable `text_body` in body/hybrid modes.",
    )
    p_analysis.add_argument(
        "--normalized-token-cap",
        type=int,
        default=40,
        help="Token cap for normalized robustness track.",
    )

    p_all = sub.add_parser("run-all", help="Collect data then run analysis pipeline.")
    p_all.add_argument(
        "--fetch-news-body",
        action="store_true",
        help="Decode RSS links and scrape full article bodies during collection.",
    )
    p_all.add_argument(
        "--news-body-timeout",
        type=int,
        default=15,
        help="HTTP timeout (seconds) for article body fetching.",
    )
    p_all.add_argument(
        "--media-text-mode",
        choices=MEDIA_TEXT_MODES,
        default="hybrid",
        help="Media text mode used by run_all_analysis.py after collection.",
    )
    p_all.add_argument(
        "--media-body-min-tokens",
        type=int,
        default=80,
        help="Minimum token threshold for usable `text_body` in body/hybrid modes.",
    )
    p_all.add_argument(
        "--normalized-token-cap",
        type=int,
        default=40,
        help="Token cap for normalized robustness track.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        return validate(
            target=args.target,
            media_text_mode=args.media_text_mode,
            fetch_news_body=args.fetch_news_body,
        )
    if args.command == "run-analysis":
        return run_analysis(
            media_text_mode=args.media_text_mode,
            media_body_min_tokens=args.media_body_min_tokens,
            normalized_token_cap=args.normalized_token_cap,
        )
    if args.command == "run-all":
        return run_all(
            media_text_mode=args.media_text_mode,
            media_body_min_tokens=args.media_body_min_tokens,
            normalized_token_cap=args.normalized_token_cap,
            fetch_news_body=args.fetch_news_body,
            news_body_timeout=args.news_body_timeout,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
