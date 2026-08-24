# Agentic VLM Starter

A deliberately small vision-language application. The UI lets the user choose GPT (OpenAI) or
Gemini (Google), choose a suggested model or enter a custom model ID, and analyze either an image
or video.

GPT image analysis uses the OpenAI Responses API and can decide to call two safe local tools:

- `get_image_metadata` for exact dimensions, format, and non-sensitive EXIF fields
- `get_dominant_colors` for a calculated palette

The UI shows the final answer, latency/token metrics, and the tool trace. Runs are appended to
`logs/runs.jsonl`; image bytes and API credentials are not logged.

Image and video previews are constrained to the display area configured by `VLM_PREVIEW_WIDTH`
and `VLM_PREVIEW_HEIGHT`, using contain scaling. This only changes the browser preview. Images
are analyzed at their validated source resolution; sampled video frames that exceed the configured
analysis-frame bounds are downsampled before they are sent to the model.

GPT video analysis uniformly samples ordered, timestamped frames and sends those still frames to
the selected GPT model. It does not send the raw video or interpret audio. Gemini video analysis
uploads the validated video to Gemini and includes its visual, audio, and timeline context.

## Architecture

```text
                         +-> GPT image agent -> Responses API + local tools
Gradio provider router -+-> GPT video agent -> sampled frames -> Responses API
                         +-> Gemini image agent -> Gemini Interactions API
                         +-> Gemini video agent -> native video -> Gemini Interactions API
```

This is agentic in a narrow, inspectable sense: the model chooses whether to use a tool, the
application executes only an allow-listed function, and the result is returned to the model for
its final answer.

## Set up on Windows PowerShell

```powershell
cd C:\MyWork\Tech-Work\VLM
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`, `GEMINI_API_KEY`, or both. A provider only requires its own
key. The file is ignored by Git.

## Run

```powershell
vlm-agent
```

Open the local URL printed by Gradio, select a provider and model, upload an image or video, enter
a question, and select **Analyze**. The model field includes recommended choices and accepts a
custom provider-compatible model ID. Supported videos are `.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`,
and `.webm`.

All media, UI, and concurrency limits are loaded through the single `LIMITS` object in
`src/vlm_agent/settings.py`. Override a value once in `.env`; validation, UI text, sampling,
and the Gradio server use the same value after the application restarts. See `.env.example`
for every available limit.

## Proof-of-concept media limits

The current defaults are listed in `.env.example`. Gradio's global upload boundary is calculated
as the larger of the configured image and video file-size limits, so it stays synchronized.
`VLM_MAX_VIDEO_WIDTH` and `VLM_MAX_VIDEO_HEIGHT` control which source videos are accepted.
`VLM_VIDEO_FRAME_WIDTH` and `VLM_VIDEO_FRAME_HEIGHT` independently control the maximum sampled
frame dimensions sent for analysis. Both landscape and portrait orientations are supported.

## Test

The unit tests do not make API calls:

```powershell
pytest
```

## Create an evaluation set

Copy `evals/example_manifest.json`, add your own images under `evals/images`, and replace the
placeholder expected terms. Then run:

```powershell
vlm-eval evals/my_manifest.json
```

The simple term check is intentionally only a starting point. A useful next step is a labeled
20-50 image dataset with task-specific scoring and regression thresholds.

## Important limitations

- Image content can contain prompt injection. The system instruction treats visible commands as
  untrusted, and the tool dispatcher exposes only two read-only functions.
- Visual answers can still be wrong. Do not use this starter for consequential decisions.
- Uploaded images are sent to the configured API model. Do not upload data you are not permitted
  to process.
- Gemini video analysis uploads the validated video to Google's File API for processing. GPT
  video analysis instead sends locally sampled frames to OpenAI.
- The evaluation runner checks expected terms, not semantic correctness.
