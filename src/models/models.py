import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class BaselineCNN(nn.Module):
    """
    Простой baseline CNN для 3 классов:
    paper / plastic / other
    """
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNReLU(3, 16, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),

            ConvBNReLU(32, 48, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(48, 64, kernel_size=3, stride=2, padding=1),

            ConvBNReLU(64, 96, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),

            ConvBNReLU(96, 128, kernel_size=1, stride=1, padding=0),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=2):
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = in_ch * expand_ratio
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        layers = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
        else:
            hidden_dim = in_ch

        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out


class MiniMobileNetV2(nn.Module):
    """
    Лёгкий MobileNetV2-style вариант для ESP32-CAM
    """
    def __init__(self, num_classes=3):
        super().__init__()

        self.stem = ConvBNReLU(3, 16, kernel_size=3, stride=2, padding=1)

        self.blocks = nn.Sequential(
            InvertedResidual(16, 24, stride=1, expand_ratio=2),
            InvertedResidual(24, 32, stride=2, expand_ratio=2),
            InvertedResidual(32, 48, stride=2, expand_ratio=2),
            InvertedResidual(48, 64, stride=1, expand_ratio=2),
            InvertedResidual(64, 80, stride=2, expand_ratio=2),
            InvertedResidual(80, 96, stride=1, expand_ratio=2),
        )

        self.head = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(model_name: str, num_classes: int = 3):
    """
    Фабрика моделей.
    Использование:
        model = get_model("baseline", 3)
        model = get_model("mobilenetv2", 3)
    """
    model_name = model_name.lower()

    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if model_name in ["mobilenetv2", "mini_mobilenetv2", "mobilev2"]:
        return MiniMobileNetV2(num_classes=num_classes)

    raise ValueError(f"Unknown model name: {model_name}")