"""Central application limits loaded once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _positive_float(name: str, default: float) -> float:
    """Read a positive floating-point environment setting with a useful error message."""
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number; received {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero; received {raw!r}.")
    return value


def _positive_int(name: str, default: int) -> int:
    """Read a positive integer environment setting with a useful error message."""
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; received {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero; received {raw!r}.")
    return value


@dataclass(frozen=True)
class MediaLimits:
    """Validated limits shared by upload validation, sampling, UI, and concurrency controls."""

    image_mb: float
    image_megapixels: float
    image_max_dimension: int
    video_mb: float
    video_seconds: float
    video_width: int
    video_height: int
    video_frame_width: int
    video_frame_height: int
    video_sampled_frames: int
    preview_width: int
    preview_height: int
    max_concurrent_analyses: int
    max_tool_rounds: int

    @property
    def upload_mb(self) -> float:
        """Return the largest upload limit needed by either supported media type."""
        return max(self.image_mb, self.video_mb)

    @classmethod
    def from_env(cls) -> "MediaLimits":
        """Build limits from environment variables and verify frame/source compatibility."""
        limits = cls(
            image_mb=_positive_float("VLM_MAX_IMAGE_MB", 10),
            image_megapixels=_positive_float("VLM_MAX_IMAGE_MEGAPIXELS", 20),
            image_max_dimension=_positive_int("VLM_MAX_IMAGE_DIMENSION", 8_000),
            video_mb=_positive_float("VLM_MAX_VIDEO_MB", 100),
            video_seconds=_positive_float("VLM_MAX_VIDEO_SECONDS", 60),
            video_width=_positive_int("VLM_MAX_VIDEO_WIDTH", 3_840),
            video_height=_positive_int("VLM_MAX_VIDEO_HEIGHT", 2_160),
            video_frame_width=_positive_int("VLM_VIDEO_FRAME_WIDTH", 1_920),
            video_frame_height=_positive_int("VLM_VIDEO_FRAME_HEIGHT", 1_080),
            video_sampled_frames=_positive_int("VLM_VIDEO_FRAMES", 12),
            preview_width=_positive_int("VLM_PREVIEW_WIDTH", 480),
            preview_height=_positive_int("VLM_PREVIEW_HEIGHT", 360),
            max_concurrent_analyses=_positive_int("VLM_MAX_CONCURRENT_ANALYSES", 4),
            max_tool_rounds=_positive_int("VLM_MAX_TOOL_ROUNDS", 4),
        )
        source_sides = sorted((limits.video_width, limits.video_height))
        frame_sides = sorted((limits.video_frame_width, limits.video_frame_height))
        if frame_sides[0] > source_sides[0] or frame_sides[1] > source_sides[1]:
            raise ValueError(
                "VLM_VIDEO_FRAME_WIDTH and VLM_VIDEO_FRAME_HEIGHT must fit within "
                "VLM_MAX_VIDEO_WIDTH and VLM_MAX_VIDEO_HEIGHT."
            )
        return limits


LIMITS = MediaLimits.from_env()
