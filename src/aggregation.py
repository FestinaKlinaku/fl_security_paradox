from collections import OrderedDict

import torch


def flatten_state_dict(state_dict: OrderedDict) -> torch.Tensor:
    return torch.cat([tensor.detach().flatten() for tensor in state_dict.values()])


def unflatten_like(vector: torch.Tensor, reference: OrderedDict) -> OrderedDict:
    rebuilt = OrderedDict()
    offset = 0
    for name, tensor in reference.items():
        numel = tensor.numel()
        rebuilt[name] = vector[offset : offset + numel].view_as(tensor).clone()
        offset += numel
    return rebuilt


def average_updates(updates: list[OrderedDict]) -> OrderedDict:
    result = OrderedDict()
    for key in updates[0].keys():
        result[key] = torch.stack([update[key] for update in updates], dim=0).mean(dim=0)
    return result


def trimmed_mean_updates(updates: list[OrderedDict], trim_count: int) -> OrderedDict:
    result = OrderedDict()
    for key in updates[0].keys():
        stacked = torch.stack([update[key] for update in updates], dim=0)
        if trim_count > 0 and stacked.shape[0] > 2 * trim_count:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_count:-trim_count]
            result[key] = trimmed.mean(dim=0)
        else:
            result[key] = stacked.mean(dim=0)
    return result


def pairwise_distances(vectors: list[torch.Tensor]) -> torch.Tensor:
    num_vectors = len(vectors)
    distances = torch.zeros((num_vectors, num_vectors), dtype=torch.float32)
    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            dist = torch.norm(vectors[i] - vectors[j], p=2)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


def krum_scores(vectors: list[torch.Tensor], num_malicious: int) -> torch.Tensor:
    distances = pairwise_distances(vectors)
    neighbor_count = max(1, len(vectors) - num_malicious - 2)
    scores = []
    for i in range(len(vectors)):
        nearest = torch.sort(distances[i])[0][1 : neighbor_count + 1]
        scores.append(nearest.sum())
    return torch.tensor(scores)


def krum_update(updates: list[OrderedDict], num_malicious: int) -> tuple[OrderedDict, int]:
    vectors = [flatten_state_dict(update) for update in updates]
    scores = krum_scores(vectors, num_malicious)
    winner = int(torch.argmin(scores).item())
    return updates[winner], winner
