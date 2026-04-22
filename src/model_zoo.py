from typing import Callable

import torch.nn as nn
from torchvision.models import mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small


SUPPORTED_MOBILENETS: dict[str, Callable[[], nn.Module]] = {
    "mobilenet": lambda: mobilenet_v2(weights=None),
    "mobilenet_v2": lambda: mobilenet_v2(weights=None),
    "mobilenet_v2_w0_65": lambda: mobilenet_v2(weights=None, width_mult=0.65),
    "mobilenet_v2_w0_40": lambda: mobilenet_v2(weights=None, width_mult=0.40),
    "mobilenet_v2_w0_28": lambda: mobilenet_v2(weights=None, width_mult=0.28),
    "mobilenet_v3_small": lambda: mobilenet_v3_small(weights=None),
    "mobilenet_v3_small_reduced": lambda: mobilenet_v3_small(
        weights=None, reduced_tail=True
    ),
    "mobilenet_v3_large": lambda: mobilenet_v3_large(weights=None),
}


def list_supported_models() -> list[str]:
    return list(SUPPORTED_MOBILENETS.keys())


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name not in SUPPORTED_MOBILENETS:
        available = ", ".join(list_supported_models())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    model = SUPPORTED_MOBILENETS[model_name]()
    replace_classifier_head(model, num_classes)
    return model


def replace_classifier_head(model: nn.Module, num_classes: int) -> None:
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for idx in range(len(model.classifier) - 1, -1, -1):
            layer = model.classifier[idx]
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                model.classifier[idx] = nn.Linear(in_features, num_classes)
                return
        raise ValueError("Classifier is Sequential, but no final Linear layer was found.")

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return

    raise ValueError(
        f"Unsupported model architecture '{model.__class__.__name__}' "
        "for automatic classifier replacement."
    )
