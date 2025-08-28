from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager

import pandas as pd
from nlp_pipeline import train_autopipeline, TextConfig, load_model


# ------------------------- tiny timer ------------------------- #
@contextmanager
def tic(label: str):
    """Lightweight timer context: prints elapsed seconds for `label`."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[TIMER] {label}: {dt:.2f}s")


# ------------------------- CLI ------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse minimal arguments for a single, full-feature run."""
    p = argparse.ArgumentParser(description="Train/eval full NLP pipeline with timers.")
    p.add_argument("--csv", type=str, required=True, help="Path to training CSV.")
    p.add_argument("--target", type=str, default="", help="Target column (omit/empty to auto-detect).")
    p.add_argument("--artifacts", type=str, default="artifacts", help="Artifacts output dir.")
    p.add_argument("--clean", action="store_true", help="Delete artifacts dir before run.")
    p.add_argument("--cv", type=int, default=3, help="CV folds (default: 3).")
    p.add_argument("--n-iter", type=int, default=10, help="Randomized search iterations (default: 10).")
    return p.parse_args()


# ------------------------- main ------------------------- #
def main() -> None:
    """Single-path, full-feature training & evaluation."""
    args = parse_args()

    # start total runtime timer
    t_start = time.perf_counter()

    artifacts_dir = args.artifacts
    model_path = os.path.join(artifacts_dir, "model.pkl")

    # Clean artifacts if requested
    if args.clean and os.path.exists(artifacts_dir):
        shutil.rmtree(artifacts_dir)
    os.makedirs(artifacts_dir, exist_ok=True)

    # Full features ON, TF–IDF auto-size (max_features=0)
    text_cfg = TextConfig(
        use_lemma=True,
        add_pos_ner=True,
        add_basic_stats=True,
        max_features=10000,
        word_ngram_range=(1, 1),
    )

    # Banner
    print("=== RUN CONFIG (FULL) ===")
    print(f"CSV:         {args.csv}")
    print(f"Target:      {args.target or '(auto-detect)'}")
    print(f"Artifacts:   {artifacts_dir}  (clean={args.clean})")
    print(f"TextConfig:  lemma={text_cfg.use_lemma}, posner={text_cfg.add_pos_ner}, "
          f"basic_stats={text_cfg.add_basic_stats}, max_features={text_cfg.max_features}, "
          f"word_ngrams={text_cfg.word_ngram_range}")
    print(f"Search:      n_iter={args.n_iter}, cv={args.cv}")
    print("=========================\n")

    # Persist run settings for the dashboard
    run_cfg = {
        "mode": "FULL",
        "csv": args.csv,
        "target": args.target or None,
        "text": {
            "use_lemma": text_cfg.use_lemma,
            "add_pos_ner": text_cfg.add_pos_ner,
            "add_basic_stats": text_cfg.add_basic_stats,
            "max_features": text_cfg.max_features,
            "word_ngram_range": text_cfg.word_ngram_range,
        },
        "search": {"n_iter": args.n_iter, "cv": args.cv},
        "artifacts_dir": artifacts_dir,
    }
    with open(os.path.join(artifacts_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    # Train + CV
    with tic("Train + CV"):
        fitted, metrics, (X_test, y_test) = train_autopipeline(
            csv_path=args.csv,
            target=(args.target or None),
            text_cfg=text_cfg,
            n_iter=args.n_iter,
            cv=args.cv,
            n_jobs=1,
            verbose=1,
            save_dir=artifacts_dir,
        )

    # Report
    with tic("Report metrics"):
        print("Metrics:", metrics)

    # Quick sanity predict (fitted)
    sample = fitted.get("example_row", {})
    if sample:
        with tic("Predict (fitted) one sample"):
            pred = fitted["pipeline"].predict(pd.DataFrame([sample]))
        print("Sample prediction (fitted):", pred)

    # Save & reload
    with tic("Save model"):
        fitted["save"](model_path)

    with tic("Load model"):
        loaded = load_model(model_path)

    if sample:
        with tic("Predict (loaded) one sample"):
            pred2 = loaded["pipeline"].predict(pd.DataFrame([sample]))
        print("Sample prediction (loaded):", pred2)

    # total runtime
    total_minutes = (time.perf_counter() - t_start) / 60.0
    print(f"\n[TOTAL TIME] Training + evaluation completed in {total_minutes:.2f} minutes.")

    print("\nArtifacts written to:", artifacts_dir)
    print("Model saved to:", model_path)


if __name__ == "__main__":
    main()
    sys.exit(0)
