# Ogaal ASR

![Ogaal ASR Overview](docs/figures/ogaal_asr_overview.png)

**Ogaal ASR** is a Somali automatic speech recognition system from Ogaal Labs. It is a fine-tuned `openai/whisper-large-v3` checkpoint trained on approximately 72.1 hours of Somali speech for practical transcription workflows.

**Model:** [Ogaal-Labs/Ogaal-ASR](https://huggingface.co/Ogaal-Labs/Ogaal-ASR)

## Key capabilities

- local CPU or GPU inference
- individual file and folder transcription
- browser demo for recording or uploading audio
- developer-ready Transformers integration
- Somali-focused speech-to-text output

## Published Somali results

- validation WER: `0.2166`
- validation CER: `0.1054`
- test WER: `0.2278`
- test CER: `0.1186`

## Training overview

- base model: [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3)
- architecture: `WhisperForConditionalGeneration`
- training rows: `39,604`
- training audio: approximately `72.1` hours
- epochs: `6`
- primary language: Somali (`so`)

A private Ogaal Labs collection contributed roughly 5,000 curated prompts recorded by 19 speakers across varied genders, accents, and speaking styles. English was not part of the training objective.

## Runtime requirements

- Python `3.10+`
- `ffmpeg` on the system path
- local model files downloaded from Hugging Face

## Quick start

```bash
git clone https://github.com/Ogaal-Labs/Ogaal-ASR.git
cd Ogaal-ASR
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone https://huggingface.co/Ogaal-Labs/Ogaal-ASR model
```

Transcribe one file:

```bash
python scripts/infer_somali_asr.py \
  --audio-path /path/to/audio.wav \
  --model-dir model
```

Run the local browser demo:

```bash
python scripts/web_demo.py --host 127.0.0.1 --port 7861 --model-dir model
```

## Transformers usage

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="Ogaal-Labs/Ogaal-ASR",
)
result = pipe("test.wav")
print(result["text"])
```

## Intended use

- Somali speech transcription
- local and privacy-sensitive transcription workflows
- developer integration for Somali voice products
- evaluation and benchmarking on Somali audio

## Limitations

- designed for Somali, not general multilingual transcription
- English was not part of the training objective
- accuracy may vary with accents, noise, microphones, domains, and speaking styles
- users should independently evaluate the model before high-stakes deployment

## Documentation

- [Technical book](docs/TECHNICAL_BOOK.md)
- [Model scope](docs/MODEL_SCOPE.md)
- [Contributing](CONTRIBUTING.md)
- [Hugging Face model card](https://huggingface.co/Ogaal-Labs/Ogaal-ASR)

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE). The published model repository uses the same license. The base `openai/whisper-large-v3` model is also distributed under Apache-2.0.

## Ogaal Labs

Ogaal Labs builds local datasets and practical AI tools for Somali and African communities.

- Website: https://ogaallabs.com/
- Hugging Face: https://huggingface.co/Ogaal-Labs/Ogaal-ASR
