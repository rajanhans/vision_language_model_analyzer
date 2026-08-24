"""Gradio UI for the starter VLM agent."""

from __future__ import annotations

import json
import os

import gradio as gr

from .agent import VLMAgent
from .gemini_agent import GeminiImageAgent, GeminiVideoAgent
from .run_logging import log_run
from .settings import LIMITS
from .video_agent import VideoVLMAgent


GPT_PROVIDER = "GPT (OpenAI)"
GEMINI_PROVIDER = "Gemini (Google)"
GPT_MODELS = ["gpt-5.6", "gpt-5.6-terra", "gpt-5.6-luna"]
GEMINI_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]


def default_model(provider: str) -> str:
    if provider == GPT_PROVIDER:
        return os.getenv("VLM_OPENAI_MODEL", os.getenv("VLM_MODEL", "gpt-5.6"))
    if provider == GEMINI_PROVIDER:
        return os.getenv("VLM_GEMINI_MODEL", "gemini-3.1-pro-preview")
    raise ValueError("Select GPT (OpenAI) or Gemini (Google).")


def update_model_options(provider: str):
    """Show useful model choices while still allowing a custom model ID."""
    choices = GPT_MODELS if provider == GPT_PROVIDER else GEMINI_MODELS
    return gr.update(choices=choices, value=default_model(provider))


def select_media(media_type: str):
    """Show only the uploader for the selected media type."""
    return (
        gr.update(visible=media_type == "Image", value=None),
        gr.update(visible=media_type == "Video", value=None),
    )


def build_agent(provider: str, media_type: str, model: str):
    """Create the provider-specific agent for the selected media type."""
    selected_model = model.strip() or default_model(provider)
    if provider == GPT_PROVIDER and media_type == "Image":
        return VLMAgent(model=selected_model)
    if provider == GPT_PROVIDER and media_type == "Video":
        return VideoVLMAgent(model=selected_model)
    if provider == GEMINI_PROVIDER and media_type == "Image":
        return GeminiImageAgent(model=selected_model)
    if provider == GEMINI_PROVIDER and media_type == "Video":
        return GeminiVideoAgent(model=selected_model)
    if media_type not in {"Image", "Video"}:
        raise ValueError("Select Image or Video.")
    raise ValueError("Select GPT (OpenAI) or Gemini (Google).")


def analyze(
    media_type: str,
    provider: str,
    image_path: str | None,
    video_path: str | None,
    question: str,
    detail: str,
    model: str,
):
    media_path = image_path if media_type == "Image" else video_path
    if not media_path:
        return f"Please upload a {media_type.lower()}.", "[]", ""
    try:
        result = build_agent(provider, media_type, model).run(media_path, question, detail)
        payload = result.to_dict()
        log_run(
            question,
            {"provider": provider, "media_type": media_type.lower(), **payload},
        )
        metrics = (
            f"Provider: {provider} · Model: {result.model} · "
            f"Latency: {result.latency_seconds:.3f}s · "
            f"Tokens: {result.usage.get('total_tokens') or 'not reported'}"
        )
        return result.answer, json.dumps(result.tool_trace, indent=2), metrics
    except Exception as exc:
        return f"Error: {exc}", "[]", ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Agentic VLM Starter") as demo:
        gr.Markdown(
            "# Agentic VLM Starter\n"
            "Choose GPT or Gemini, select image or video, upload the media, and ask a "
            "question. GPT video analysis uses ordered sampled frames; Gemini analyzes "
            "the uploaded video with its visual, audio, and timeline context."
        )
        gr.Markdown(
            "**Proof-of-concept limits**  \n"
            f"Images: up to **{LIMITS.image_mb:g} MB**, "
            f"**{LIMITS.image_megapixels:g} megapixels**, and "
            f"**{LIMITS.image_max_dimension:,} px** on either side.  \n"
            f"Videos: up to **{LIMITS.video_mb:g} MB**, "
            f"**{LIMITS.video_seconds:g} seconds**, and "
            f"**{LIMITS.video_width}x{LIMITS.video_height}** source resolution "
            "(or the portrait equivalent). GPT sampled frames are resized to fit "
            f"**{LIMITS.video_frame_width}x{LIMITS.video_frame_height}** and use at most "
            f"**{LIMITS.video_sampled_frames} frames**. Gemini receives the validated "
            "video directly."
        )
        with gr.Row():
            with gr.Column(scale=1):
                media_type = gr.Radio(
                    choices=["Image", "Video"], value="Image", label="Interpret"
                )
                provider = gr.Radio(
                    choices=[GPT_PROVIDER, GEMINI_PROVIDER],
                    value=GPT_PROVIDER,
                    label="Provider",
                )
                image = gr.Image(
                    type="filepath",
                    label="Image",
                    width=LIMITS.preview_width,
                    height=LIMITS.preview_height,
                    elem_classes=["media-preview"],
                )
                video = gr.Video(
                    label="Video",
                    visible=False,
                    width=LIMITS.preview_width,
                    height=LIMITS.preview_height,
                    elem_classes=["media-preview"],
                )
                question = gr.Textbox(
                    label="Question",
                    placeholder="What is happening in this image or video?",
                    lines=3,
                )
                with gr.Row():
                    detail = gr.Dropdown(
                        choices=["auto", "low", "high"],
                        value="auto",
                        label="Visual detail",
                    )
                    model = gr.Dropdown(
                        choices=GPT_MODELS,
                        value=default_model(GPT_PROVIDER),
                        allow_custom_value=True,
                        label="Model",
                    )
                submit = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                answer = gr.Markdown(label="Answer")
                metrics = gr.Markdown()
                trace = gr.Code(label="Agent tool trace", language="json")

        gr.Examples(
            examples=[
                ["Describe the scene and identify any uncertainty."],
                ["What are the dominant colors, and what percentage does each occupy?"],
                ["What are the exact pixel dimensions and file format?"],
                ["Transcribe visible text. Mark anything unreadable as [unclear]."],
            ],
            inputs=[question],
        )
        media_type.change(select_media, media_type, [image, video])
        provider.change(update_model_options, provider, model)
        analyze_inputs = [media_type, provider, image, video, question, detail, model]
        submit.click(analyze, analyze_inputs, [answer, trace, metrics])
        question.submit(analyze, analyze_inputs, [answer, trace, metrics])
    return demo


def main() -> None:
    build_app().queue(default_concurrency_limit=LIMITS.max_concurrent_analyses).launch(
        max_file_size=f"{LIMITS.upload_mb:g}mb",
        max_threads=LIMITS.max_concurrent_analyses,
        css=(
            f".media-preview {{ max-width: {LIMITS.preview_width}px !important; "
            "margin-inline: auto; } "
            ".media-preview img, .media-preview video { "
            f"max-height: {LIMITS.preview_height}px !important; "
            "object-fit: contain !important; }"
        ),
    )


if __name__ == "__main__":
    main()
