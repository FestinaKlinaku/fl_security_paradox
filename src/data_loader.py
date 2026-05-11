import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


def load_femnist_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_client_datasets(path: str | Path) -> dict[str, TensorDataset]:
    data = load_femnist_json(path)
    client_datasets: dict[str, TensorDataset] = {}

    for user in data["users"]:
        user_x = torch.tensor(data["user_data"][user]["x"], dtype=torch.float32)
        user_y = torch.tensor(data["user_data"][user]["y"], dtype=torch.long)
        client_datasets[user] = TensorDataset(user_x, user_y)

    return client_datasets


def build_client_loaders(
    path: str | Path, batch_size: int = 16, shuffle: bool = True
) -> dict[str, DataLoader]:
    datasets = build_client_datasets(path)
    return {
        user: DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        for user, dataset in datasets.items()
    }


def build_global_test_loader(path: str | Path, batch_size: int = 64) -> DataLoader:
    data = load_femnist_json(path)
    all_x = []
    all_y = []

    for user in data["users"]:
        all_x.extend(data["user_data"][user]["x"])
        all_y.extend(data["user_data"][user]["y"])

    dataset = TensorDataset(
        torch.tensor(all_x, dtype=torch.float32),
        torch.tensor(all_y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
