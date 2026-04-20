from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ROIConfig:
    x: int
    y: int
    w: int
    h: int

    def as_xywh(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def validate(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError("ROI width and height must be positive.")
        if self.x < 0 or self.y < 0:
            raise ValueError("ROI x and y must be non-negative.")


@dataclass(frozen=True, slots=True)
class PresenceDetectorConfig:
    roi: ROIConfig = field(default_factory=lambda: ROIConfig(x=0, y=0, w=224, h=224))
    processing_size: tuple[int, int] = (64, 64)
    blur_kernel: int = 5
    diff_threshold: int = 18
    min_foreground_ratio: float = 0.03
    enter_frames: int = 2
    exit_frames: int = 5
    morph_kernel_size: int = 3
    morph_open_iterations: int = 1
    morph_close_iterations: int = 1
    background_alpha: float = 0.02
    # Keep the adaptive-background code path in Python, but default to a fixed
    # calibrated background because that is the safer baseline for the future
    # ESP runtime.
    use_adaptive_background: bool = False
    input_color_order: str = "bgr"
    min_brightness: float = 20.0
    max_brightness: float = 235.0

    def __post_init__(self) -> None:
        self.roi.validate()

        width, height = self.processing_size
        if width <= 0 or height <= 0:
            raise ValueError("processing_size must contain positive width and height.")
        if self.blur_kernel <= 0 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd number.")
        if self.diff_threshold < 0:
            raise ValueError("diff_threshold must be non-negative.")
        if not 0.0 <= self.min_foreground_ratio <= 1.0:
            raise ValueError("min_foreground_ratio must be in [0, 1].")
        if self.enter_frames <= 0 or self.exit_frames <= 0:
            raise ValueError("enter_frames and exit_frames must be positive.")
        if self.morph_kernel_size <= 0:
            raise ValueError("morph_kernel_size must be positive.")
        if self.background_alpha < 0.0 or self.background_alpha > 1.0:
            raise ValueError("background_alpha must be in [0, 1].")
        if self.input_color_order.lower() not in {"bgr", "rgb"}:
            raise ValueError("input_color_order must be either 'bgr' or 'rgb'.")
        if self.min_brightness < 0.0 or self.max_brightness > 255.0:
            raise ValueError("Brightness bounds must stay in [0, 255].")
        if self.min_brightness > self.max_brightness:
            raise ValueError("min_brightness must not exceed max_brightness.")


@dataclass(frozen=True, slots=True)
class RuntimePipelineConfig:
    classifier_roi: ROIConfig | None = None
    classifier_input_size: tuple[int, int] = (96, 96)
    collect_frames_for_selection: int = 3
    cooldown_frames: int = 8

    def __post_init__(self) -> None:
        width, height = self.classifier_input_size
        if width <= 0 or height <= 0:
            raise ValueError("classifier_input_size must contain positive width and height.")
        if self.collect_frames_for_selection <= 0:
            raise ValueError("collect_frames_for_selection must be positive.")
        if self.cooldown_frames < 0:
            raise ValueError("cooldown_frames must be non-negative.")
        if self.classifier_roi is not None:
            self.classifier_roi.validate()


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    detector: PresenceDetectorConfig = field(default_factory=PresenceDetectorConfig)
    pipeline: RuntimePipelineConfig = field(default_factory=RuntimePipelineConfig)
