"""Video validation, metadata, and representative-frame sampling."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from .settings import LIMITS

SUPPORTED_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class VideoMetadata:
    """Technical properties collected from a video container and stream."""

    filename: str
    format: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    file_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to the JSON-compatible form used in traces and prompts."""
        return asdict(self)


@dataclass(frozen=True)
class SampledFrame:
    """One chronological frame prepared for transmission to a vision model."""

    timestamp_seconds: float
    frame_number: int
    width: int
    height: int
    was_resized: bool
    data_url: str


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Read container metadata using OpenCV without decoding every video frame."""
    path = Path(video_path).resolve()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("The uploaded video could not be opened.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("The uploaded video has invalid or unreadable metadata.")
        return VideoMetadata(
            filename=path.name,
            format=path.suffix.lower().lstrip(".") or "unknown",
            width=width,
            height=height,
            fps=round(fps, 3),
            frame_count=frame_count,
            duration_seconds=round(frame_count / fps, 3),
            file_size_bytes=path.stat().st_size,
        )
    finally:
        capture.release()


def validate_video(
    video_path: str | Path,
    max_video_mb: float | None = None,
    max_duration_seconds: float | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
) -> tuple[Path, VideoMetadata]:
    """Validate file type, size, resolution, duration, and basic stream decodability."""
    path = Path(video_path).resolve()
    if not path.is_file():
        raise ValueError("Please upload a video file.")
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video type. Supported extensions: {allowed}.")
    max_video_mb = min(
        float(max_video_mb) if max_video_mb is not None else LIMITS.video_mb,
        LIMITS.video_mb,
    )
    max_duration_seconds = min(
        float(max_duration_seconds)
        if max_duration_seconds is not None
        else LIMITS.video_seconds,
        LIMITS.video_seconds,
    )
    max_width = min(
        int(max_width) if max_width is not None else LIMITS.video_width,
        LIMITS.video_width,
    )
    max_height = min(
        int(max_height) if max_height is not None else LIMITS.video_height,
        LIMITS.video_height,
    )
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_video_mb:
        raise ValueError(f"Video is {size_mb:.1f} MB; the limit is {max_video_mb:g} MB.")

    metadata = get_video_metadata(path)
    source_sides = sorted((metadata.width, metadata.height))
    limit_sides = sorted((max_width, max_height))
    if source_sides[0] > limit_sides[0] or source_sides[1] > limit_sides[1]:
        raise ValueError(
            f"Video resolution is {metadata.width}x{metadata.height}; the maximum accepted "
            f"source resolution is {max_width}x{max_height} landscape or "
            f"{max_height}x{max_width} portrait. Resize the video and try again."
        )
    if metadata.duration_seconds > max_duration_seconds:
        raise ValueError(
            f"Video is {metadata.duration_seconds:.1f}s; the limit is "
            f"{max_duration_seconds:g}s. Trim the video and try again."
        )
    return path, metadata


def frame_to_data_url(frame: Any, jpeg_quality: int = 85) -> str:
    """Encode one OpenCV BGR frame as a bounded-quality JPEG data URL."""
    success, encoded = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, max(40, min(jpeg_quality, 95))]
    )
    if not success:
        raise ValueError("A sampled video frame could not be encoded.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def resize_frame(
    frame: Any,
    max_width: int | None = None,
    max_height: int | None = None,
) -> Any:
    """Downsize a frame to an orientation-aware box while preserving its aspect ratio."""
    source_height, source_width = frame.shape[:2]
    target_width = max_width or LIMITS.video_frame_width
    target_height = max_height or LIMITS.video_frame_height
    if source_height > source_width:
        target_width, target_height = target_height, target_width
    scale = min(target_width / source_width, target_height / source_height, 1.0)
    if scale >= 1.0:
        return frame
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    return cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def sample_video_frames(
    video_path: str | Path,
    frame_count: int = 12,
    jpeg_quality: int = 85,
) -> list[SampledFrame]:
    """Uniformly sample, resize, timestamp, and encode readable frames across a video."""
    metadata = get_video_metadata(video_path)
    requested = max(1, min(int(frame_count), LIMITS.video_sampled_frames))
    sample_total = min(requested, metadata.frame_count)
    # Include the first and last frames so the model can compare the full available timeline.
    if sample_total == 1:
        indices = [0]
    else:
        indices = sorted(
            {
                round(index * (metadata.frame_count - 1) / (sample_total - 1))
                for index in range(sample_total)
            }
        )

    capture = cv2.VideoCapture(str(video_path))
    frames: list[SampledFrame] = []
    try:
        if not capture.isOpened():
            raise ValueError("The uploaded video could not be opened for frame sampling.")
        for frame_number in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = capture.read()
            if not success:
                continue
            source_height, source_width = frame.shape[:2]
            frame = resize_frame(frame)
            frame_height, frame_width = frame.shape[:2]
            frames.append(
                SampledFrame(
                    timestamp_seconds=round(frame_number / metadata.fps, 3),
                    frame_number=frame_number,
                    width=frame_width,
                    height=frame_height,
                    was_resized=(frame_width, frame_height)
                    != (source_width, source_height),
                    data_url=frame_to_data_url(frame, jpeg_quality),
                )
            )
    finally:
        capture.release()

    if not frames:
        raise ValueError("No readable frames could be extracted from the video.")
    return frames


VIDEO_TOOLS = [
    {
        "type": "function",
        "name": "get_video_metadata",
        "description": (
            "Get exact video duration, frame rate, frame count, dimensions, format, and file "
            "size. Use this when the user asks for technical video facts."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    }
]


def call_video_tool(name: str, arguments_json: str, video_path: str | Path) -> str:
    """Validate arguments, dispatch the allow-listed metadata tool, and serialize its result."""
    json.loads(arguments_json or "{}")
    if name != "get_video_metadata":
        raise ValueError(f"Unknown video tool: {name}")
    return json.dumps(get_video_metadata(video_path).to_dict(), ensure_ascii=False)
