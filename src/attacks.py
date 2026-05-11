from collections import OrderedDict

import torch


def apply_poisoning(
    update: OrderedDict, attack_type: str = "none", strength: float = 1.0
) -> OrderedDict:
    if attack_type == "none":
        return OrderedDict((name, tensor.clone()) for name, tensor in update.items())

    attacked = OrderedDict()
    for name, tensor in update.items():
        if attack_type == "aggressive":
            attacked[name] = tensor * (-strength)
        elif attack_type == "stealthy":
            attacked[name] = tensor + (torch.sign(tensor) * strength * 0.05)
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")
    return attacked
