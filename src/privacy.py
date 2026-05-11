from collections import OrderedDict

import torch


def epsilon_to_noise_scale(epsilon: float, clipping_norm: float = 1.0) -> float:
    epsilon = max(epsilon, 1e-6)
    return clipping_norm / epsilon


def clip_update(update: OrderedDict, max_norm: float = 1.0) -> OrderedDict:
    total_norm = torch.sqrt(
        sum(torch.sum(tensor.detach() ** 2) for tensor in update.values())
    )
    scale = min(1.0, max_norm / (total_norm.item() + 1e-12))
    return OrderedDict((name, tensor * scale) for name, tensor in update.items())


def add_dp_noise(
    update: OrderedDict, epsilon: float, clipping_norm: float = 1.0
) -> OrderedDict:
    noise_scale = epsilon_to_noise_scale(epsilon, clipping_norm)
    clipped = clip_update(update, max_norm=clipping_norm)
    noised = OrderedDict()

    for name, tensor in clipped.items():
        noise = torch.normal(mean=0.0, std=noise_scale, size=tensor.shape)
        noised[name] = tensor + noise

    return noised
