import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    parser.add_argument(
        "--aggregations",
        nargs="+",
        default=["fedavg", "krum", "trimmed_mean"],
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["stealthy"],
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--malicious-fraction", type=float, default=0.2)
    parser.add_argument("--attack-strength", type=float, default=3.0)
    parser.add_argument("--train-path", default="femnist/data/sample_small/train/sampled_small_train.json")
    parser.add_argument("--test-path", default="femnist/data/sample_small/test/sampled_small_test.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results/sweeps")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = []
    for aggregation in args.aggregations:
        for attack in args.attacks:
            for epsilon in args.epsilons:
                safe_epsilon = str(epsilon).replace(".", "_")
                output_path = output_dir / f"{aggregation}_{attack}_eps_{safe_epsilon}.json"
                command = [
                    sys.executable,
                    "main.py",
                    "--train-path",
                    args.train_path,
                    "--test-path",
                    args.test_path,
                    "--aggregation",
                    aggregation,
                    "--epsilon",
                    str(epsilon),
                    "--attack-type",
                    attack,
                    "--attack-strength",
                    str(args.attack_strength),
                    "--rounds",
                    str(args.rounds),
                    "--local-epochs",
                    str(args.local_epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--malicious-fraction",
                    str(args.malicious_fraction),
                    "--device",
                    args.device,
                    "--output",
                    str(output_path),
                ]
                print("Running:", " ".join(command))
                subprocess.run(command, check=True)
                run_manifest.append(
                    {
                        "aggregation": aggregation,
                        "attack": attack,
                        "epsilon": epsilon,
                        "output": str(output_path),
                    }
                )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
