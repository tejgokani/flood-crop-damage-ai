"""v2 entry point.

    python run.py pilot            # does synthetic data help? one model, both conditions
    python run.py full             # all three models, synthetic on
    python run.py report           # rebuild tables from reports/

Target: macro-F1 >= 0.70 on the four severity classes, measured out-of-fold on real tiles only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent
sys.path.insert(0, str(V1 / "src"))
sys.path.insert(0, str(ROOT))

from fcda2.data.dataset import class_counts, load_real_pool  # noqa: E402
from fcda2.train.runner import MODE_FOR_MODEL, run  # noqa: E402

from fcda.models.registry import DISPLAY_NAMES, build_model  # noqa: E402
from fcda.train.loop import pick_device  # noqa: E402

DATA = V1 / "data"
REPORTS = ROOT / "reports"
MODELS = ["yolo12_unet", "cnn_lstm", "effnet_attention"]

#: The number to beat. v1 reached 0.462; the always-Healthy baseline scores 0.703 *accuracy*
#: and a macro-F1 of 0.206, which is why macro-F1 is the target and accuracy is not.
TARGET_MACRO_F1 = 0.70


def _build(name: str):
    in_ch = 6 if MODE_FOR_MODEL[name] == "change" else 3
    if name == "effnet_attention":
        return lambda: build_model(name, pretrained=True, in_channels=in_ch)
    return lambda: build_model(name, pretrained=False)


def _header(pool):
    print(f"real pool: {len(pool)} tiles  {class_counts(pool)}")
    print(f"target: macro-F1 >= {TARGET_MACRO_F1}  (v1 best 0.462; always-Healthy macro-F1 0.206)")


def cmd_pilot(args) -> int:
    """Does procedural synthetic data help? Same folds, one variable."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    pool = load_real_pool(DATA)
    _header(pool)
    dev = pick_device(args.device)
    name = args.model
    print(f"\nPILOT: {DISPLAY_NAMES[name]} | {args.folds} folds | {args.max_minutes} min/fold | {dev}")

    results = {}
    for use_syn in (False, True):
        print(f"\n{'=' * 70}\n  {'WITH' if use_syn else 'WITHOUT'} synthetic\n{'=' * 70}")
        results[use_syn] = run(
            name, _build(name), pool, DATA, size=args.size, n_folds=args.folds,
            use_synthetic=use_syn, per_class_target=args.per_class,
            batch_size=args.batch, max_minutes=args.max_minutes, epochs=args.epochs,
            device=dev, reports_dir=REPORTS, verbose=True,
        )

    print(f"\n{'=' * 70}\n  PILOT RESULT\n{'=' * 70}")
    rows = ["| Condition | macro-F1 | Accuracy | Within-1 | Severe F1 | Flood IoU | Gap |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for use_syn, label in ((False, "Real only"), (True, "Real + synthetic")):
        o = results[use_syn].oof
        rows.append(
            f"| {label} | **{o.get('macro_f1', 0):.3f}** | {o.get('accuracy', 0):.3f} "
            f"| {o.get('within_one_accuracy', 0):.3f} "
            f"| {o.get('per_class_f1', {}).get('Severe', 0):.3f} "
            f"| {o.get('iou', 0):.3f} | {results[use_syn].mean_gap:+.3f} |")
    delta = results[True].oof.get("macro_f1", 0) - results[False].oof.get("macro_f1", 0)
    print("\n".join(rows))
    print(f"\nDelta (synthetic - real only): {delta:+.3f} macro-F1")
    print("Verdict:", "synthetic HELPS" if delta > 0.01 else
          ("synthetic HURTS" if delta < -0.01 else "no meaningful effect"))
    (REPORTS / "pilot.json").write_text(json.dumps(
        {"model": name, "delta_macro_f1": delta,
         "real_only": results[False].to_dict(), "with_synthetic": results[True].to_dict()}, indent=2))
    return 0


def cmd_full(args) -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pool = load_real_pool(DATA)
    _header(pool)
    dev = pick_device(args.device)
    out = {}
    for name in (args.models or MODELS):
        print(f"\n{'=' * 70}\n  {DISPLAY_NAMES[name]}\n{'=' * 70}")
        r = run(name, _build(name), pool, DATA, size=args.size, n_folds=args.folds,
                use_synthetic=not args.no_synthetic, per_class_target=args.per_class,
                batch_size=args.batch, max_minutes=args.max_minutes, epochs=args.epochs,
                device=dev, reports_dir=REPORTS, verbose=True)
        out[name] = r.to_dict()
        o = r.oof
        hit = "MET" if o.get("macro_f1", 0) >= TARGET_MACRO_F1 else "not met"
        print(f"\n  {DISPLAY_NAMES[name]}: macro-F1 {o.get('macro_f1', 0):.3f} -> target {hit}")
    (REPORTS / "v2_results.json").write_text(json.dumps(out, indent=2))
    return 0


def cmd_report(args) -> int:
    from fcda2.eval.report import build

    print(build(REPORTS))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="v2", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--folds", type=int, default=3)
        sp.add_argument("--size", type=int, default=192)
        sp.add_argument("--batch", type=int, default=8)
        sp.add_argument("--max-minutes", type=float, default=25.0)
        sp.add_argument("--epochs", type=int, default=30)
        sp.add_argument("--per-class", type=int, default=600)
        sp.add_argument("--device", default="auto")

    a = sub.add_parser("pilot")
    common(a)
    a.add_argument("--model", default="yolo12_unet", choices=MODELS)
    a.set_defaults(func=cmd_pilot)

    b = sub.add_parser("full")
    common(b)
    b.add_argument("--models", nargs="*", choices=MODELS)
    b.add_argument("--no-synthetic", action="store_true")
    b.set_defaults(func=cmd_full)

    c = sub.add_parser("report")
    c.set_defaults(func=cmd_report)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
