"""Command line interface.

    fcda download --tier T1_tiny     fetch one tier's tiles
    fcda smoke                       offline end-to-end run (no network, used by CI)
    fcda train --progressive         walk the tier ladder with all five hybrids
    fcda tabular                     the CSV/SMOTE sequence on Indian crop statistics
    fcda report                      regenerate figures and tables from reports/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data.tiers import TIERS, get_tier
from .models.registry import all_model_names

DATA_ROOT = Path("data")
REPORTS = Path("reports")


def _cmd_download(args) -> int:
    from .data.download import download_india_csv
    from .data.etci import build_index

    tier = get_tier(args.tier)
    print(f"Fetching tier {tier.name}: {tier.n_tiles} tiles (~{tier.approx_disk_mb:.0f} MB)")
    records = build_index(DATA_ROOT, tier.n_tiles, progress=True)
    print(f"  {len(records)} tiles ready")
    csv = download_india_csv(DATA_ROOT)
    print(f"  Indian crop statistics: {csv} ({csv.stat().st_size // 1024} KB)")
    return 0


def _cmd_smoke(args) -> int:
    """Offline proof that every step runs. No network, no GPU, under two minutes."""
    from .data.tiers import get_tier
    from .pipeline import run_image_pipeline

    tier = get_tier("T0_smoke")
    res = run_image_pipeline(
        tier,
        DATA_ROOT,
        model_names=args.models or ["effnet_attention"],
        reports_dir=REPORTS,
        offline=True,
        max_minutes_per_model=args.max_minutes,
        gan_epochs=args.gan_epochs,
        device=args.device,
        verbose=True,
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "smoke.json").write_text(json.dumps(res.to_dict(), indent=2))
    ok = all(m.ok for m in res.models)
    print(f"\nSmoke run {'PASSED' if ok else 'FAILED'} in {res.seconds:.0f}s")
    return 0 if ok else 1


def _cmd_train(args) -> int:
    from .pipeline import run_image_pipeline, run_progressive

    models = args.models or all_model_names()
    REPORTS.mkdir(parents=True, exist_ok=True)

    if args.progressive:
        out = run_progressive(
            DATA_ROOT, REPORTS, models,
            start_tier=args.tier, offline=args.offline,
            max_minutes_per_model=args.max_minutes, gan_epochs=args.gan_epochs,
            budget_minutes=args.budget, device=args.device, verbose=True,
            resume=args.resume,
        )
        print(f"\nFinished at tier {out['scaling']['final_tier']}: {out['scaling']['stopped_because']}")
        return 0

    res = run_image_pipeline(
        get_tier(args.tier), DATA_ROOT, models, REPORTS,
        offline=args.offline, max_minutes_per_model=args.max_minutes,
        gan_epochs=args.gan_epochs, device=args.device, verbose=True,
    )
    (REPORTS / "results.json").write_text(json.dumps({"tiers": [res.to_dict()]}, indent=2))
    return 0


def _cmd_cv(args) -> int:
    from .pipeline import run_cv_pipeline

    REPORTS.mkdir(parents=True, exist_ok=True)
    run_cv_pipeline(
        get_tier(args.tier), DATA_ROOT, args.models or all_model_names(), REPORTS,
        offline=args.offline, n_folds=args.folds,
        max_minutes_per_fold=args.max_minutes, device=args.device, verbose=True,
    )
    return 0


def _cmd_tabular(args) -> int:
    from .data.download import download_india_csv
    from .pipeline import run_tabular_pipeline

    csv = download_india_csv(DATA_ROOT)
    out = run_tabular_pipeline(csv, REPORTS)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "tabular.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {REPORTS / 'tabular.json'}")
    return 0


def _cmd_report(args) -> int:
    from .eval.report import build_all

    build_all(REPORTS)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fcda", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="fetch one dataset tier")
    d.add_argument("--tier", default="T1_tiny", choices=[t.name for t in TIERS])
    d.set_defaults(func=_cmd_download)

    s = sub.add_parser("smoke", help="offline end-to-end run")
    s.add_argument("--models", nargs="*", choices=all_model_names())
    s.add_argument("--max-minutes", type=float, default=2.0)
    s.add_argument("--gan-epochs", type=int, default=2)
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=_cmd_smoke)

    t = sub.add_parser("train", help="train the hybrids")
    t.add_argument("--tier", default="T1_tiny", choices=[x.name for x in TIERS])
    t.add_argument("--models", nargs="*", choices=all_model_names())
    t.add_argument("--progressive", action="store_true", help="walk the tier ladder")
    t.add_argument("--offline", action="store_true", help="use synthetic data")
    t.add_argument("--max-minutes", type=float, default=25.0)
    t.add_argument("--gan-epochs", type=int, default=30)
    t.add_argument("--budget", type=float, default=45.0, help="per-tier budget for the capacity gate")
    t.add_argument("--resume", action="store_true",
                   help="carry forward tiers already present in reports/results.json")
    t.add_argument("--device", default="auto")
    t.set_defaults(func=_cmd_train)

    cv = sub.add_parser("cv", help="k-fold cross validation (headline evaluation)")
    cv.add_argument("--tier", default="T3_medium", choices=[x.name for x in TIERS])
    cv.add_argument("--models", nargs="*", choices=all_model_names())
    cv.add_argument("--folds", type=int, default=5)
    cv.add_argument("--max-minutes", type=float, default=14.0, help="wall-clock cap per fold")
    cv.add_argument("--offline", action="store_true")
    cv.add_argument("--device", default="auto")
    cv.set_defaults(func=_cmd_cv)

    tb = sub.add_parser("tabular", help="the CSV/SMOTE sequence")
    tb.set_defaults(func=_cmd_tabular)

    r = sub.add_parser("report", help="regenerate figures and tables")
    r.set_defaults(func=_cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
