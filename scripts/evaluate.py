# Downloads history for a city, compares the models, and saves the results.
#
#   python scripts/evaluate.py --city Delhi
#   python scripts/evaluate.py --demo        (fake data, no internet)

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from mauka.config import CITIES  # noqa: E402
from mauka.data import DataError, load_history, synthetic_series  # noqa: E402
from mauka.model import evaluate_models  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Delhi", choices=sorted(CITIES))
    parser.add_argument("--start", default="2022-08-01")
    parser.add_argument("--end", default=(date.today() - timedelta(days=7)).isoformat())
    parser.add_argument("--models", default="persistence,climatology,logistic,hgb,tabpfn")
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--max-test", type=int, default=3000, help="use fewer test rows to go faster")
    parser.add_argument("--demo", action="store_true", help="use fake data")
    args = parser.parse_args()

    if args.demo:
        df = synthetic_series("2022-08-01", "2025-12-31")
        label = "synthetic"
    else:
        lat, lon = CITIES[args.city]
        try:
            df = load_history(args.city, lat, lon, args.start, args.end)
        except DataError as e:
            sys.exit(f"Could not load data: {e}")
        label = args.city

    out = evaluate_models(df, models=args.models.split(","),
                          max_train=args.max_train, max_test=args.max_test)
    out["city"] = label
    out["data"] = "synthetic" if args.demo else "open-meteo"

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"metrics_{label.lower()}.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"\nCity: {label} | train rows: {out['n_train']} | test rows: {out['n_test']}")
    print(f"Test period: {out['test_period'][0]} to {out['test_period'][1]}")
    print(f"Share of good hours in the test period: {out['test_base_rate']:.1%}\n")
    print(pd.DataFrame(out["results"]).round(3).to_string(index=False))
    for name, reason in out["skipped"].items():
        print(f"\nSkipped {name}: {reason}")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
