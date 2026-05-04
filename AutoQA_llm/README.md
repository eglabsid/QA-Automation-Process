# AutoQA_llm

`AutoQA_llm` is the LLM-assisted QA workspace for gameplay video analysis, bug triage, Reddit/video collection, and sound-effect log extraction.

## Run

Install the component dependencies from the repository root:

```sh
pip install -r AutoQA_llm/requirements.txt
```

Launch the desktop UI:

```sh
python AutoQA_llm/main.py
```

If the local display server needs an explicit Qt backend:

```sh
QT_QPA_PLATFORM=wayland python AutoQA_llm/main.py
```

## Sound-Effect Log Extraction

The video analysis flow extracts timestamped sound-effect logs from real video audio before optional Gemini analysis. The default extractor is local and deterministic: it uses FFmpeg to create WAV audio, then computes energy, direction, confidence, and type labels from the signal.

For offline extraction, make sure `ffmpeg` is on `PATH`, or set one of these environment variables:

```sh
export COMPRESSO_FFMPEG_PATH="/path/to/ffmpeg"
export FFMPEG_PATH="/path/to/ffmpeg"
```

Gemini analysis is optional. When an API key is configured, the local sound logs are added to the prompt as evidence:

```sh
export GEMINI_API_KEY="your-api-key"
```

Expected sound timelines can be loaded from JSON to validate missing, desynced, mismatched, or unexpected sounds. See `docs/sound_expected_events.example.json` in the repository root.
