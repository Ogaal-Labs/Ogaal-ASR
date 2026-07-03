# Ogaal ASR

![Ogaal ASR Overview](docs/figures/ogaal_asr_overview.png)

`Ogaal ASR` is the public Somali ASR release from Ogaal Labs.

The Hugging Face model page is being finalized and will be linked here once it is live.

## Overview

This repository packages:

- a local Somali ASR inference CLI
- a browser demo for recording or uploading audio
- product documentation for the released model
- a clean open-source entry point for developers

## Highlights

- Somali automatic speech recognition model based on `openai/whisper-large-v3`
- trained on roughly `72.1` hours of Somali speech
- private prompt-based multi-speaker Somali data collection
- local CLI workflow for file and folder transcription
- local browser demo for recording or uploading audio

## Why Ogaal ASR Was Built

Ogaal Labs builds local datasets and practical AI tools for Somali and African communities. `Ogaal ASR` was intentionally trained as a Somali ASR model so the training objective stayed aligned with Somali speech recognition and Somali data development.

English was not part of the training objective for this release.

## Data Overview

The training data used for this release totals roughly `72.1` hours of Somali speech.

A private Ogaal Labs collection pipeline contributed a core part of that effort through roughly `5,000` curated prompts recorded by `19` speakers across varied genders, accents, and speaking styles.

## Runtime Requirements

- Python `3.10+`
- `ffmpeg` available on the system path
- local model files placed in `model/` or passed through `--model-dir`

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the model into `model/` yourself or pass `--model-dir` explicitly.

CLI example:

```bash
python scripts/infer_somali_asr.py \
  --audio-path /path/to/audio.wav \
  --model-dir /path/to/model_repo
```

Browser demo:

```bash
python scripts/web_demo.py --host 127.0.0.1 --port 7861 --model-dir /path/to/model_repo
```

Common developer commands:

```bash
make check
make demo MODEL_DIR=/path/to/model_repo
make infer MODEL_DIR=/path/to/model_repo AUDIO=/path/to/audio.wav
```

## Somali Metrics

- validation WER: `0.2166`
- validation CER: `0.1054`
- test WER: `0.2278`
- test CER: `0.1186`

## Repository Layout

```text
Ogaal-ASR/
├── docs/
│   ├── figures/
│   ├── MODEL_SCOPE.md
│   ├── PUBLICATION_CHECKLIST.md
│   ├── README.md
│   └── TECHNICAL_BOOK.md
├── examples/
├── metadata/
├── scripts/
├── .github/
├── CHANGELOG.md
├── CONTRIBUTING.md
├── Makefile
├── PUSHING.md
└── requirements.txt
```

## Ogaal Labs

- organization: `Ogaal Labs`
- website: `https://ogaallabs.com/`

## Documentation

- technical book: `docs/TECHNICAL_BOOK.md`
- model scope: `docs/MODEL_SCOPE.md`
- publication checklist: `docs/PUBLICATION_CHECKLIST.md`
- documentation index: `docs/README.md`

## Development

- run `make check` before pushing
- keep large model files out of Git and use Hugging Face for weights
- keep local inference outputs untracked
