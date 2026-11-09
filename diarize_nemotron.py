import argparse
import json
import os
import sys
import traceback
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
MODEL_ID = "nvidia/Nemotron-3-Diarization"
MODEL_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"


class NoSpeech(ValueError):
    """Known per-input failure, not a broken resident model."""


class InvalidAudio(ValueError):
    """Known malformed/empty audio, detected outside model inference."""


def verify_coverage(duration, frame_seconds, valid_frames, output_frames):
    if abs(valid_frames * frame_seconds - duration) > 2 * frame_seconds or not valid_frames <= output_frames <= valid_frames + 7:
        raise RuntimeError(f"Incomplete audio: {duration:.2f}s, {valid_frames} input frames, {output_frames} output frames")


def load_model(local_files_only=False):
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("An AMD ROCm GPU is required")
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                               local_files_only=local_files_only)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=local_files_only).to("cuda").eval()
    return processor, model


def diarize_one(processor, model, audio, output):
    """All inputs, logits and speaker identities are request-local, not cached."""
    import torch
    from transformers.audio_utils import load_audio

    audio, output = Path(audio), Path(output)
    if audio.resolve() == output.resolve():
        raise InvalidAudio("Output must not overwrite the input audio")
    rate = processor.feature_extractor.sampling_rate
    try:
        audio = load_audio(str(audio), sampling_rate=rate)
    except (OSError, ValueError, EOFError) as error:
        raise InvalidAudio(str(error)) from error
    if not len(audio):
        raise InvalidAudio("Input audio is empty")
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
        raise NoSpeech("No speaker segments returned")
    lines = []
    for seg in segments:
        begin, end = seg["Start"], min(seg["End"], duration)
        if begin < end:
            lines.append(f"{begin}\t{end}\tspeaker_{seg['Speaker']}\n")
    if not lines:
        raise NoSpeech("No segments within audio duration")
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    tmp.replace(output)


def _worker_loop(source, protocol):
    """TSV in, JSONL out; return only at EOF or a fatal worker error."""
    from model_workers import ModelInputError, _wire_path

    def send(**fields):
        protocol.write(json.dumps({"v": 1, **fields}, ensure_ascii=True, allow_nan=False) + "\n")
        protocol.flush()

    try:
        processor, model = load_model(local_files_only=True)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        send(ready=False, service="diarization", pid=os.getpid(), fatal=True, error=str(error) or type(error).__name__)
        return 1
    send(ready=True, service="diarization", pid=os.getpid())
    previous = 0
    while True:
        # Windows paths are limited to 32767 UTF-16 units each. 256 KiB permits
        # two such UTF-8 paths and the sequence, but bounds malformed requests.
        line = source.readline(256 * 1024 + 1)
        if not line:
            return 0
        sequence = 0
        try:
            if len(line) > 256 * 1024 or not line.endswith("\n"):
                raise RuntimeError("Incomplete or oversized worker request")
            fields = line[:-1].split("\t")
            if (len(fields) != 3 or not fields[0].isascii() or not fields[0].isdecimal()
                    or int(fields[0]) <= previous):
                raise RuntimeError("Invalid worker request sequence/schema")
            sequence = int(fields[0])
            previous = sequence
            audio, output = map(_wire_path, fields[1:])
            if Path(audio).resolve() == Path(output).resolve():
                raise InvalidAudio("Output must not overwrite the input audio")
            try:
                with wave.open(audio, "rb") as stream:
                    if not stream.getnframes():
                        raise InvalidAudio("Input audio is empty")
            except (OSError, EOFError, wave.Error) as error:
                raise InvalidAudio(str(error)) from error
            diarize_one(processor, model, audio, output)
        except (NoSpeech, InvalidAudio, ModelInputError) as error:
            send(id=sequence, ok=False, fatal=False, error=str(error) or type(error).__name__)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            send(id=sequence, ok=False, fatal=True, error=str(error) or type(error).__name__)
            return 1
        else:
            send(id=sequence, ok=True, result=None)
        # diarize_one's frame (including inputs/logits) is gone before the next
        # request. Nemotron forward creates its own SpeakerCache each time.


def _worker_main():
    # Reserve a protocol descriptor BEFORE importing torch/transformers. Redirect
    # both Python and native-library stdout; C/ROCm diagnostics cannot enter JSONL.
    sys.stdout.flush()
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    with os.fdopen(protocol_fd, "w", encoding="utf-8", newline="\n", buffering=1) as protocol:
        sys.stdin.reconfigure(encoding="utf-8", newline="\n")
        return _worker_loop(sys.stdin, protocol)


def main(argv=None):
    parser = argparse.ArgumentParser(description="ROCm Nemotron speaker diarization (no transcription)")
    parser.add_argument("audio", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--worker", action="store_true", help="Serve serial requests using cached models only")
    args = parser.parse_args(argv)
    if args.worker:
        if args.audio is not None or args.output is not None:
            parser.error("--worker does not accept positional paths")
        return _worker_main()
    if args.audio is None or args.output is None:
        parser.error("audio and output are required")
    if args.audio.resolve() == args.output.resolve():
        raise ValueError("Output must not overwrite the input audio")
    processor, model = load_model()
    diarize_one(processor, model, args.audio, args.output)
    return 0


if __name__ == "__main__":
    # Windows ROCm can hang during Python shutdown. In worker mode this happens
    # only at outer EOF/fatal, never after a successfully handled input.
    try:
        code = main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
