"""Safe, deterministic local tools available to the VLM agent."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from .settings import LIMITS


def validate_image(
    image_path: str | Path,
    max_image_mb: float | None = None,
    max_megapixels: float | None = None,
    max_dimension: int | None = None,
) -> Path:
    """Validate that a path is a readable image within the configured size limit."""
    path = Path(image_path).resolve()
    if not path.is_file():
        raise ValueError("Please upload an image file.")

    max_image_mb = min(
        float(max_image_mb) if max_image_mb is not None else LIMITS.image_mb,
        LIMITS.image_mb,
    )
    max_megapixels = min(
        float(max_megapixels) if max_megapixels is not None else LIMITS.image_megapixels,
        LIMITS.image_megapixels,
    )
    max_dimension = min(
        int(max_dimension) if max_dimension is not None else LIMITS.image_max_dimension,
        LIMITS.image_max_dimension,
    )
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_image_mb:
        raise ValueError(f"Image is {size_mb:.1f} MB; the limit is {max_image_mb:g} MB.")

    try:
        with Image.open(path) as image:
            megapixels = image.width * image.height / 1_000_000
            if megapixels > max_megapixels:
                raise ValueError(
                    f"Image is {megapixels:.1f} megapixels; the limit is "
                    f"{max_megapixels:g} megapixels."
                )
            if max(image.width, image.height) > max_dimension:
                raise ValueError(
                    f"Image dimensions are {image.width}x{image.height}; neither dimension may "
                    f"exceed {max_dimension:,} pixels."
                )
            image.verify()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc
    return path


def image_to_data_url(image_path: str | Path) -> str:
    """Encode a local image as an API-compatible data URL."""
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def get_image_metadata(image_path: str | Path) -> dict[str, Any]:
    """Return non-sensitive technical image metadata."""
    path = Path(image_path)
    with Image.open(path) as image:
        exif: dict[str, str] = {}
        for key, value in image.getexif().items():
            name = ExifTags.TAGS.get(key, str(key))
            if name in {"GPSInfo", "MakerNote", "UserComment"}:
                continue
            exif[name] = str(value)[:200]

        return {
            "filename": path.name,
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "megapixels": round(image.width * image.height / 1_000_000, 3),
            "file_size_bytes": path.stat().st_size,
            "exif": exif,
        }


def get_dominant_colors(image_path: str | Path, count: int = 5) -> dict[str, Any]:
    """Estimate dominant colors using a small quantized copy of the image."""
    count = max(1, min(int(count), 10))
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.thumbnail((256, 256))
        quantized = image.quantize(colors=count, method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette() or []
        color_counts = sorted(quantized.getcolors() or [], reverse=True)

    total = sum(pixel_count for pixel_count, _ in color_counts) or 1
    colors = []
    for pixel_count, palette_index in color_counts[:count]:
        offset = palette_index * 3
        rgb = tuple(palette[offset : offset + 3])
        colors.append(
            {
                "hex": "#" + "".join(f"{channel:02X}" for channel in rgb),
                "rgb": rgb,
                "share_percent": round(pixel_count / total * 100, 1),
            }
        )
    return {"colors": colors, "method": "median-cut approximation"}


TOOLS = [
    {
        "type": "function",
        "name": "get_image_metadata",
        "description": (
            "Get exact image dimensions, file format, color mode, size, and safe EXIF fields. "
            "Use this when the question requires technical facts rather than visual estimates."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_dominant_colors",
        "description": (
            "Calculate an approximate dominant-color palette for the uploaded image. "
            "Use this for palette, branding, or color-composition questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of colors to return, from 1 through 10.",
                    "minimum": 1,
                    "maximum": 10,
                }
            },
            "required": ["count"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def call_image_tool(name: str, arguments_json: str, image_path: str | Path) -> str:
    """Dispatch an allow-listed image tool and serialize its result."""
    arguments = json.loads(arguments_json or "{}")
    if name == "get_image_metadata":
        result = get_image_metadata(image_path)
    elif name == "get_dominant_colors":
        result = get_dominant_colors(image_path, count=arguments["count"])
    else:
        raise ValueError(f"Unknown tool: {name}")
    return json.dumps(result, ensure_ascii=False)
