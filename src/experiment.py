import argparse
import json
import random
from pathlib import Path

from src.aggregation import (
    average_updates,
    flatten_state_dict,
    krum_scores,
    krum_update,
    trimmed_mean_updates,
)
from src.attacks import apply_poisoning
from src.data_loader import build_client_loaders, build_global_test_loader
from src.federated import apply_global_update, evaluate_model, train_local_model
from src.model import FemnistMLP
from src.privacy import add_dp_noise


def select_malicious_clients(client_ids: list[str], fraction: float, seed: int) -> set[str]:
    rng = random.Random(seed)
    count = max(1, int(len(client_ids) * fraction))
    return set(rng.sample(client_ids, count))


def detect_malicious_clients(
    updates: list,
    client_ids: list[str],
    method: str,
    num_malicious: int,
) -> set[str]:
    vectors = [flatten_state_dict(update) for update in updates]

    if method == "krum":
        scores = krum_scores(vectors, num_malicious)
        ranked = sorted(zip(client_ids, scores.tolist()), key=lambda item: item[1], reverse=True)
        return {client_id for client_id, _ in ranked[:num_malicious]}

    center = sum(vectors) / len(vectors)
    distances = [(client_id, (vector - center).norm().item()) for client_id, vector in zip(client_ids, vectors)]
    ranked = sorted(distances, key=lambda item: item[1], reverse=True)
    return {client_id for client_id, _ in ranked[:num_malicious]}


def aggregate_updates(updates: list, method: str, num_malicious: int):
    if method == "fedavg":
        return average_updates(updates)
    if method == "trimmed_mean":
        return trimmed_mean_updates(updates, trim_count=num_malicious)
    if method == "krum":
        update, _ = krum_update(updates, num_malicious=num_malicious)
        return update
    raise ValueError(f"Unsupported aggregation method: {method}")


def compute_detection_metrics(
    predicted_malicious: set[str], actual_malicious: set[str], all_clients: list[str]
) -> dict[str, float]:
    true_positive = len(predicted_malicious & actual_malicious)
    false_positive = len(predicted_malicious - actual_malicious)
    false_negative = len(actual_malicious - predicted_malicious)
    true_negative = len(set(all_clients) - predicted_malicious - actual_malicious)

    detection_accuracy = (true_positive + true_negative) / max(len(all_clients), 1)
    false_positive_rate = false_positive / max(
        len(set(all_clients) - actual_malicious), 1
    )
    true_positive_rate = true_positive / max(len(actual_malicious), 1)
    false_negative_rate = false_negative / max(len(actual_malicious), 1)

    return {
        "detection_accuracy": detection_accuracy,
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def run_experiment(args) -> dict:
    train_loaders = build_client_loaders(args.train_path, batch_size=args.batch_size)
    test_loader = build_global_test_loader(args.test_path, batch_size=args.batch_size)
    client_ids = sorted(train_loaders.keys())
    malicious_clients = select_malicious_clients(
        client_ids, fraction=args.malicious_fraction, seed=args.seed
    )
    num_malicious = len(malicious_clients)

    model = FemnistMLP()
    history = []

    for round_idx in range(1, args.rounds + 1):
        round_updates = []

        for client_id in client_ids:
            update = train_local_model(
                global_model=model,
                dataloader=train_loaders[client_id],
                device=args.device,
                epochs=args.local_epochs,
                learning_rate=args.learning_rate,
            )

            if client_id in malicious_clients:
                update = apply_poisoning(
                    update, attack_type=args.attack_type, strength=args.attack_strength
                )

            if args.epsilon is not None:
                update = add_dp_noise(update, epsilon=args.epsilon, clipping_norm=1.0)

            round_updates.append(update)

        predicted_malicious = detect_malicious_clients(
            round_updates, client_ids, args.aggregation, num_malicious
        )
        aggregated_update = aggregate_updates(
            round_updates, method=args.aggregation, num_malicious=num_malicious
        )
        apply_global_update(model, aggregated_update)

        eval_metrics = evaluate_model(model, test_loader, device=args.device)
        detection_metrics = compute_detection_metrics(
            predicted_malicious, malicious_clients, client_ids
        )

        round_result = {
            "round": round_idx,
            "aggregation": args.aggregation,
            "epsilon": args.epsilon,
            "attack_type": args.attack_type,
            "malicious_clients": sorted(malicious_clients),
            **eval_metrics,
            **detection_metrics,
        }
        history.append(round_result)
        print(json.dumps(round_result))

    return {
        "config": vars(args),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        default="femnist/data/sample_small/train/sampled_small_train.json",
    )
    parser.add_argument(
        "--test-path",
        default="femnist/data/sample_small/test/sampled_small_test.json",
    )
    parser.add_argument("--aggregation", choices=["fedavg", "trimmed_mean", "krum"], default="fedavg")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--attack-type", choices=["none", "aggressive", "stealthy"], default="none")
    parser.add_argument("--attack-strength", type=float, default=3.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--malicious-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/latest_run.json")
    args = parser.parse_args()

    results = run_experiment(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
