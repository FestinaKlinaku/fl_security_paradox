from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn


def train_local_model(
    global_model: nn.Module,
    dataloader,
    device: str,
    epochs: int = 1,
    learning_rate: float = 0.01,
) -> OrderedDict:
    local_model = deepcopy(global_model).to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = local_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    return compute_model_update(global_model.state_dict(), local_model.state_dict())


def compute_model_update(
    base_state: OrderedDict, updated_state: OrderedDict
) -> OrderedDict:
    return OrderedDict(
        (name, updated_state[name].detach().cpu() - base_state[name].detach().cpu())
        for name in base_state.keys()
    )


def apply_global_update(model: nn.Module, update: OrderedDict) -> None:
    current_state = model.state_dict()
    for name in current_state.keys():
        current_state[name] = current_state[name] + update[name].to(current_state[name].device)
    model.load_state_dict(current_state)


def evaluate_model(model: nn.Module, dataloader, device: str) -> dict[str, float]:
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == batch_y).sum().item()
            total_examples += batch_x.size(0)

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }
