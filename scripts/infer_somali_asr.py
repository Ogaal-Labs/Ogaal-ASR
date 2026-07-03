#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from transformers import WhisperForConditionalGeneration, WhisperProcessor


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TARGET_SAMPLE_RATE = 16000
DEFAULT_MODEL_DIR = PACKAGE_ROOT / "model"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs"
SUPPORTED_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".webm"}
DEFAULT_WORKFLOW = "whisper_large_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the finished Somali Whisper large-v3 ASR model on one file or a folder.")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--audio-path", type=Path)
    target_group.add_argument("--audio-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--decoder", choices=[DEFAULT_WORKFLOW], default=DEFAULT_WORKFLOW)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--long-audio-seconds", type=float, default=18.0)
    parser.add_argument("--max-chunk-seconds", type=float, default=18.0)
    parser.add_argument("--chunk-overlap-seconds", type=float, default=1.0)
    parser.add_argument("--frame-ms", type=int, default=30)
    parser.add_argument("--hop-ms", type=int, default=10)
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--min-silence-ms", type=int, default=350)
    parser.add_argument("--speech-pad-ms", type=int, default=150)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def load_audio(audio_path: Path) -> tuple[torch.Tensor, int]:
    ffmpeg_command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        decoded = subprocess.run(ffmpeg_command, check=True, capture_output=True)
        waveform = np.frombuffer(decoded.stdout, dtype=np.float32)
        if waveform.size == 0:
            raise RuntimeError("ffmpeg returned empty audio")
        return torch.from_numpy(waveform.copy()), TARGET_SAMPLE_RATE
    except (FileNotFoundError, subprocess.CalledProcessError, RuntimeError):
        sample_rate, waveform = wavfile.read(audio_path)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if np.issubdtype(waveform.dtype, np.integer):
            max_value = max(abs(np.iinfo(waveform.dtype).min), np.iinfo(waveform.dtype).max)
            waveform = waveform.astype(np.float32) / float(max_value)
        else:
            waveform = waveform.astype(np.float32)
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform_tensor = torchaudio.functional.resample(waveform_tensor, sample_rate, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE
        return waveform_tensor.squeeze(0).contiguous(), sample_rate


def detect_speech_regions(
    audio: torch.Tensor,
    sample_rate: int,
    *,
    frame_ms: int,
    hop_ms: int,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> list[tuple[int, int]]:
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop_len = max(1, int(sample_rate * hop_ms / 1000))
    if audio.numel() <= frame_len:
        return [(0, int(audio.numel()))]

    windows = audio.unfold(0, frame_len, hop_len)
    rms = torch.sqrt(torch.clamp(windows.pow(2).mean(dim=1), min=1e-10)).cpu().numpy()
    rms_db = 20.0 * np.log10(rms + 1e-12)
    noise_db = float(np.percentile(rms_db, 15))
    peak_db = float(np.percentile(rms_db, 95))
    threshold_db = max(-45.0, min(noise_db + 12.0, peak_db - 8.0))
    active = rms_db > threshold_db

    min_speech_frames = max(1, math.ceil(min_speech_ms / hop_ms))
    min_silence_frames = max(1, math.ceil(min_silence_ms / hop_ms))
    speech_pad_samples = int(sample_rate * speech_pad_ms / 1000)

    regions: list[tuple[int, int]] = []
    start_frame: int | None = None
    silence_run = 0

    for frame_index, is_active in enumerate(active):
        if is_active:
            if start_frame is None:
                start_frame = frame_index
            silence_run = 0
            continue
        if start_frame is None:
            continue
        silence_run += 1
        if silence_run < min_silence_frames:
            continue
        end_frame = frame_index - silence_run + 1
        if end_frame - start_frame >= min_speech_frames:
            start_sample = max(0, start_frame * hop_len - speech_pad_samples)
            end_sample = min(int(audio.numel()), end_frame * hop_len + frame_len + speech_pad_samples)
            regions.append((start_sample, end_sample))
        start_frame = None
        silence_run = 0

    if start_frame is not None:
        end_frame = len(active)
        if end_frame - start_frame >= min_speech_frames:
            start_sample = max(0, start_frame * hop_len - speech_pad_samples)
            end_sample = int(audio.numel())
            regions.append((start_sample, end_sample))

    if not regions:
        return [(0, int(audio.numel()))]

    merged: list[list[int]] = []
    for start_sample, end_sample in regions:
        if not merged or start_sample - merged[-1][1] > speech_pad_samples:
            merged.append([start_sample, end_sample])
        else:
            merged[-1][1] = max(merged[-1][1], end_sample)
    return [(start, end) for start, end in merged]


def split_fixed_windows(
    audio_length: int,
    sample_rate: int,
    *,
    max_chunk_seconds: float,
    chunk_overlap_seconds: float,
) -> list[tuple[int, int]]:
    max_chunk_samples = int(max_chunk_seconds * sample_rate)
    overlap_samples = int(chunk_overlap_seconds * sample_rate)
    stride = max(1, max_chunk_samples - overlap_samples)
    windows = []
    start = 0
    while start < audio_length:
        end = min(audio_length, start + max_chunk_samples)
        windows.append((start, end))
        if end >= audio_length:
            break
        start += stride
    return windows


def build_chunks(
    audio: torch.Tensor,
    sample_rate: int,
    *,
    long_audio_seconds: float,
    max_chunk_seconds: float,
    chunk_overlap_seconds: float,
    frame_ms: int,
    hop_ms: int,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    duration_seconds = float(audio.numel() / sample_rate)
    if duration_seconds <= long_audio_seconds:
        return [
            {
                "index": 0,
                "start_sample": 0,
                "end_sample": int(audio.numel()),
                "start_seconds": 0.0,
                "end_seconds": duration_seconds,
                "audio": audio,
            }
        ], {
            "used_chunking": False,
            "used_vad": False,
            "region_count": 1,
        }

    regions = detect_speech_regions(
        audio,
        sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )

    max_chunk_samples = int(max_chunk_seconds * sample_rate)
    chunks: list[dict[str, Any]] = []
    current_start: int | None = None
    current_end: int | None = None
    chunk_index = 0

    for region_start, region_end in regions:
        if current_start is None:
            current_start, current_end = region_start, region_end
            continue
        assert current_end is not None
        if region_end - current_start <= max_chunk_samples:
            current_end = region_end
            continue
        chunks.append(
            {
                "index": chunk_index,
                "start_sample": current_start,
                "end_sample": current_end,
                "start_seconds": current_start / sample_rate,
                "end_seconds": current_end / sample_rate,
                "audio": audio[current_start:current_end],
            }
        )
        chunk_index += 1
        current_start, current_end = region_start, region_end

    if current_start is not None and current_end is not None:
        chunks.append(
            {
                "index": chunk_index,
                "start_sample": current_start,
                "end_sample": current_end,
                "start_seconds": current_start / sample_rate,
                "end_seconds": current_end / sample_rate,
                "audio": audio[current_start:current_end],
            }
        )

    final_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_audio = chunk["audio"]
        if chunk_audio.numel() <= max_chunk_samples:
            final_chunks.append(chunk)
            continue
        for window_start, window_end in split_fixed_windows(
            int(chunk_audio.numel()),
            sample_rate,
            max_chunk_seconds=max_chunk_seconds,
            chunk_overlap_seconds=chunk_overlap_seconds,
        ):
            absolute_start = chunk["start_sample"] + window_start
            absolute_end = chunk["start_sample"] + window_end
            final_chunks.append(
                {
                    "index": len(final_chunks),
                    "start_sample": absolute_start,
                    "end_sample": absolute_end,
                    "start_seconds": absolute_start / sample_rate,
                    "end_seconds": absolute_end / sample_rate,
                    "audio": audio[absolute_start:absolute_end],
                }
            )

    return final_chunks, {
        "used_chunking": True,
        "used_vad": True,
        "region_count": len(regions),
    }


def collect_audio_paths(audio_dir: Path, recursive: bool) -> list[Path]:
    iterator = audio_dir.rglob("*") if recursive else audio_dir.glob("*")
    paths = [path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(paths)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def ensure_model(model_dir: Path) -> tuple[WhisperProcessor, WhisperForConditionalGeneration, torch.device]:
    processor = WhisperProcessor.from_pretrained(model_dir)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=dtype, low_cpu_mem_usage=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    model.generation_config.language = "so"
    model.generation_config.task = "transcribe"
    return processor, model, device


def transcribe_audio(
    *,
    audio_path: Path,
    model_dir: Path,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: torch.device,
    decoder_name: str,
    decoder,
    beam_width: int,
    chunk_args: dict[str, Any],
) -> dict[str, Any]:
    del decoder
    del beam_width
    started_at = time.time()
    waveform, sample_rate = load_audio(audio_path)
    chunks, chunk_meta = build_chunks(waveform, sample_rate, **chunk_args)

    chunk_results: list[dict[str, Any]] = []
    transcript_parts: list[str] = []
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="so", task="transcribe")

    for chunk in chunks:
        features = processor.feature_extractor(
            chunk["audio"].numpy(),
            sampling_rate=sample_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        model_dtype = next(model.parameters()).dtype
        inputs = {
            key: (value.to(device=device, dtype=model_dtype) if key == "input_features" else value.to(device))
            for key, value in features.items()
        }
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
            )

        prediction_text = normalize_whitespace(
            processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        )
        transcript_parts.append(prediction_text)
        chunk_results.append(
            {
                "chunk_index": chunk["index"],
                "start_seconds": round(chunk["start_seconds"], 3),
                "end_seconds": round(chunk["end_seconds"], 3),
                "duration_seconds": round(chunk["end_seconds"] - chunk["start_seconds"], 3),
                "prediction_text": prediction_text,
            }
        )

    transcript_text = normalize_whitespace(" ".join(part for part in transcript_parts if part))
    duration_seconds = float(waveform.numel() / sample_rate)
    runtime_seconds = time.time() - started_at
    return {
        "audio_path": str(audio_path),
        "audio_name": audio_path.name,
        "decoder": decoder_name,
        "model_dir": str(model_dir),
        "sample_rate_hz": sample_rate,
        "duration_seconds": round(duration_seconds, 3),
        "used_chunking": chunk_meta["used_chunking"],
        "used_vad": chunk_meta["used_vad"],
        "vad_region_count": chunk_meta["region_count"],
        "chunk_count": len(chunk_results),
        "runtime_seconds": round(runtime_seconds, 3),
        "transcript_text": transcript_text,
        "chunks": chunk_results,
    }


def save_single_result(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result["audio_name"]).stem
    (output_dir / f"{stem}.txt").write_text(result["transcript_text"] + "\n", encoding="utf-8")
    (output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_batch_outputs(results: list[dict[str, Any]], output_dir: Path, model_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for result in results:
        save_single_result(result, output_dir)
        summary_rows.append(
            {
                "audio_name": result["audio_name"],
                "audio_path": result["audio_path"],
                "decoder": result["decoder"],
                "duration_seconds": result["duration_seconds"],
                "chunk_count": result["chunk_count"],
                "used_chunking": result["used_chunking"],
                "transcript_text": result["transcript_text"],
            }
        )

    with (output_dir / "batch_results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    with (output_dir / "batch_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "total_files": len(results),
        "decoder": results[0]["decoder"] if results else "",
        "model_dir": str(model_dir),
        "total_audio_seconds": round(sum(result["duration_seconds"] for result in results), 3),
        "total_runtime_seconds": round(sum(result["runtime_seconds"] for result in results), 3),
        "used_chunking_files": sum(int(result["used_chunking"]) for result in results),
    }
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    processor, model, device = ensure_model(args.model_dir)

    if args.audio_path is not None:
        audio_paths = [args.audio_path]
    else:
        audio_paths = collect_audio_paths(args.audio_dir, recursive=args.recursive)
        if not audio_paths:
            raise FileNotFoundError(f"No supported audio files found in {args.audio_dir}")

    chunk_args = {
        "long_audio_seconds": args.long_audio_seconds,
        "max_chunk_seconds": args.max_chunk_seconds,
        "chunk_overlap_seconds": args.chunk_overlap_seconds,
        "frame_ms": args.frame_ms,
        "hop_ms": args.hop_ms,
        "min_speech_ms": args.min_speech_ms,
        "min_silence_ms": args.min_silence_ms,
        "speech_pad_ms": args.speech_pad_ms,
    }

    results = []
    for audio_path in audio_paths:
        result = transcribe_audio(
            audio_path=audio_path,
            model_dir=args.model_dir,
            processor=processor,
            model=model,
            device=device,
            decoder_name=args.decoder,
            decoder=None,
            beam_width=0,
            chunk_args=chunk_args,
        )
        results.append(result)
        print(f"[{audio_path.name}] {result['transcript_text']}")

    if args.audio_path is not None:
        save_single_result(results[0], args.output_dir)
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    else:
        write_batch_outputs(results, args.output_dir, args.model_dir)
        print(json.dumps({"total_files": len(results), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
