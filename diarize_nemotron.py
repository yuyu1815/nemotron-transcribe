import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
MODEL_ID = "nvidia/Nemotron-3-Diarization"
MODEL_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"


def verify_coverage(duration, frame_seconds, valid_frames, output_frames):
    if abs(valid_frames * frame_seconds - duration) > 2 * frame_seconds or not valid_frames <= output_frames <= valid_frames + 7:
        raise RuntimeError(f"Incomplete audio: {duration:.2f}s, {valid_frames} input frames, {output_frames} output frames")


def main():
    parser = argparse.ArgumentParser(description="ROCm Nemotron speaker diarization (no transcription)")
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.audio.resolve() == args.output.resolve():
        raise ValueError("Output must not overwrite the input audio")

    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor
    from transformers.audio_utils import load_audio

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("An AMD ROCm GPU is required")
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to("cuda").eval()
    rate = processor.feature_extractor.sampling_rate
    audio = load_audio(str(args.audio), sampling_rate=rate)
    duration = len(audio) / rate
    inputs = processor(audio, sampling_rate=rate).to(model.device, dtype=model.dtype)
    del audio
    torch.cuda.synchronize()
    with torch.inference_mode():
        logits = model(**inputs).logits
    torch.cuda.synchronize()
    valid_frames = int(inputs.attention_mask.sum().item())
    verify_coverage(duration, processor.feature_extractor.hop_length / rate, valid_frames, logits.shape[1])
    segments = processor.extract_speaker_dict(logits.cpu(), inputs.attention_mask.cpu())[0]
    if not segments:
        raise RuntimeError("No speaker segments returned")
    lines = []
    for seg in segments:
        begin, end = seg["Start"], min(seg["End"], duration)
        if begin < end:
            lines.append(f"{begin}\t{end}\tspeaker_{seg['Speaker']}\n")
    if not lines:
        raise RuntimeError("No segments within audio duration")
    tmp = args.output.with_name(args.output.name + ".tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    tmp.replace(args.output)


if __name__ == "__main__":
    # Windows ROCm can hang during Python shutdown after successful inference.
    try:
        main()
    except SystemExit as exc:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exc.code if isinstance(exc.code, int) else 1)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
