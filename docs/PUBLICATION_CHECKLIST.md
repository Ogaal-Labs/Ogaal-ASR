# Publication Checklist

## Before Publishing

- confirm the final public model name
- confirm the Hugging Face repo id
- confirm the GitHub repo name
- confirm the license after data review
- confirm there is no sensitive local path leakage in copied docs
- confirm the model card keeps the Somali-only training position clear

## Build

- run `build_release_artifacts.py`
- inspect `artifacts/hf_model_repo/README.md`
- inspect `artifacts/github_repo/README.md`
- smoke-test the CLI with `--model-dir`
- smoke-test the browser demo locally

## Hugging Face

- create the model repo
- upload the built `hf_model_repo/` folder
- verify `README.md` renders correctly
- verify `model.safetensors` is stored with LFS
- verify the example inference snippet runs

## GitHub

- publish `artifacts/github_repo/`
- verify the repo README points to the correct Hugging Face model repo
- verify there are no large scratch outputs in the repo
- verify the demo and CLI instructions use relative paths only

## Release Messaging

- say the model was intentionally trained for Somali speech recognition
- cite the held-out Somali metrics
- say the training effort totals roughly 72 hours of Somali speech
- mention the private multi-speaker prompt collection method without exposing private data
