import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_result_files(results_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        history = payload.get("history", [])
        if not history:
            continue
        final = history[-1]
        records.append(
            {
                "file": path.name,
                "aggregation": config.get("aggregation", final.get("aggregation")),
                "attack_type": config.get("attack_type", final.get("attack_type")),
                "epsilon": float(config.get("epsilon", final.get("epsilon", 0.0))),
                "round": final.get("round"),
                "loss": final.get("loss"),
                "accuracy": final.get("accuracy"),
                "detection_accuracy": final.get("detection_accuracy"),
                "false_positive_rate": final.get("false_positive_rate"),
                "true_positive_rate": final.get("true_positive_rate"),
                "false_negative_rate": final.get("false_negative_rate"),
            }
        )
    if not records:
        raise ValueError(f"No result json files found in {results_dir}")
    return pd.DataFrame(records).sort_values(["aggregation", "attack_type", "epsilon"])


def save_plot(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for (aggregation, attack_type), group in df.groupby(["aggregation", "attack_type"]):
        group = group.sort_values("epsilon")
        label = f"{aggregation} | {attack_type}"
        plt.plot(group["epsilon"], group[metric], marker="o", label=label)

    plt.xscale("log")
    plt.xlabel("Epsilon")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} vs Epsilon")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/sweeps")
    parser.add_argument("--plots-dir", default="results/plots")
    parser.add_argument("--summary-csv", default="results/plots/summary.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_result_files(results_dir)
    df.to_csv(args.summary_csv, index=False)

    for metric in [
        "accuracy",
        "loss",
        "detection_accuracy",
        "false_positive_rate",
        "true_positive_rate",
        "false_negative_rate",
    ]:
        save_plot(df, metric, plots_dir / f"{metric}_vs_epsilon.png")

    print(f"Saved summary CSV to {args.summary_csv}")
    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
