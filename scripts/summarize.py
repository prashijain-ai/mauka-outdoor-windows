# Prints one markdown table from every results/metrics_*.json file.
#
#   python scripts/summarize.py

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    files = sorted(RESULTS_DIR.glob("metrics_*.json"))
    if not files:
        print("No results yet. Run: python scripts/evaluate.py --city <city>")
        return

    print("| City | Test period | Good hours | Model | AUC | Brier | Accuracy | F1 |")
    print("|---|---|---|---|---|---|---|---|")
    for path in files:
        data = json.loads(path.read_text())
        if data.get("data") == "synthetic":
            continue  # never publish made-up data
        period = f"{data['test_period'][0]} to {data['test_period'][1]}"
        rate = f"{data['test_base_rate']:.0%}"
        for row in data["results"]:
            print(f"| {data['city']} | {period} | {rate} | {row['model']} | "
                  f"{row['auc']:.3f} | {row['brier']:.3f} | {row['accuracy']:.3f} | {row['f1']:.3f} |")

    print("\nBest model per city (lowest Brier score):")
    for path in files:
        data = json.loads(path.read_text())
        if data.get("data") == "synthetic" or not data["results"]:
            continue
        best = min(data["results"], key=lambda r: r["brier"])
        print(f"- {data['city']}: {best['model']} (Brier {best['brier']:.3f})")


if __name__ == "__main__":
    main()
