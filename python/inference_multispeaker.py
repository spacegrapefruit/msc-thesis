"""
Inference script for multispeaker TTS models.
This script generates audio samples for the test sentences using different speakers.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def get_latest_model_checkpoint(training_dir):
    """Find the best model checkpoint in the training directory."""
    best_model_path = training_dir / "best_model.pth"
    config_path = training_dir / "config.json"
    speakers_path = training_dir / "speakers.pth"

    if not all([best_model_path.exists(), config_path.exists()]):
        return None, None, None

    return (
        str(best_model_path),
        str(config_path),
        str(speakers_path) if speakers_path.exists() else None,
    )


def load_test_sentences(config_path):
    """Load test sentences from the config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("test_sentences", [])


def run_inference(
    model_path,
    config_path,
    speakers_path,
    text,
    speaker_id,
    output_path,
    use_gpu=True,
    device=None,
    vocoder_path=None,
    vocoder_config_path=None,
):
    """Run TTS inference using the tts command."""
    cmd = [
        "tts",
        "--text",
        text,
        "--model_path",
        model_path,
        "--config_path",
        config_path,
        "--out_path",
        output_path,
    ]

    if speakers_path and os.path.exists(speakers_path):
        cmd.extend(["--speakers_file_path", speakers_path])

    if speaker_id is not None:
        cmd.extend(["--speaker_idx", speaker_id])

    # Add vocoder support
    if vocoder_path:
        cmd.extend(["--vocoder_path", vocoder_path])
        if vocoder_config_path:
            cmd.extend(["--vocoder_config_path", vocoder_config_path])

    # Add GPU support
    if use_gpu:
        if device:
            cmd.extend(["--device", device])
        else:
            cmd.append("--use_cuda")

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Generated: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating {output_path}:")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on multispeaker TTS models"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to the model directory (e.g., training_output/Tacotron2-DDC-CV-LT-Multispeaker-November-06-2025_10+04PM-6805a9d)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_output",
        help="Output directory for generated audio files",
    )
    parser.add_argument(
        "--speakers",
        type=str,
        default="F4_VP382,M3_VP460",
        help="Comma-separated list of speaker IDs to use (default: F4_VP382,M3_VP460)",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Custom text to synthesize (if not provided, uses test_sentences from config)",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="Use GPU for inference (default: True)",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU usage (use CPU instead)",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Specific device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to custom vocoder model file",
    )
    parser.add_argument(
        "--vocoder_config_path",
        type=str,
        help="Path to custom vocoder config file",
    )

    args = parser.parse_args()

    if not args.model_dir:
        # Auto-detect the latest model directory
        training_output_dir = Path("training_output")
        assert training_output_dir.exists(), "No training_output directory found"

        # Find the most recent multispeaker model
        model_dirs = list(training_output_dir.glob("Tacotron2-DDC-*-Multispeaker-*"))
        assert model_dirs, "No multispeaker model directories found in training_output"

        # Sort by modification time, get the latest
        model_dirs.sort(key=lambda x: x.stat().st_mtime)
        model_dir = model_dirs[-1]
        print(f"Auto-detected latest model: {model_dir}")
    else:
        model_dir = Path(args.model_dir)

    assert model_dir.exists(), "Model directory does not exist"

    # Get model files
    model_path, config_path, speakers_path = get_latest_model_checkpoint(model_dir)
    assert model_path is not None, "Could not find model checkpoint"

    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print(f"Speakers: {speakers_path if speakers_path else 'Not found'}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Parse speaker IDs
    speaker_ids = [s.strip() for s in args.speakers.split(",")]
    print(f"Using speakers: {speaker_ids}")

    # Configure GPU usage
    use_gpu = args.use_gpu and not args.no_gpu
    device = args.device

    if args.no_gpu:
        use_gpu = False
        device = "cpu"
        print("GPU disabled - using CPU for inference")
    elif device:
        use_gpu = True  # Force GPU usage if device is specified
        print(f"Using device: {device}")
    elif use_gpu:
        print("Using GPU for inference (CUDA)")
    else:
        print("Using CPU for inference")

    # Configure vocoder
    vocoder_path = args.vocoder_path
    vocoder_config_path = args.vocoder_config_path

    if vocoder_path:
        print(f"Using custom vocoder: {vocoder_path}")
        if vocoder_config_path:
            print(f"Vocoder config: {vocoder_config_path}")
    else:
        print("No vocoder specified - using Griffin-Lim reconstruction (lower quality)")

    # Get test sentences
    if args.text:
        test_sentences = [args.text]
        print(f"Using custom text: {args.text}")
    else:
        test_sentences = load_test_sentences(config_path)
        print(f"Loaded {len(test_sentences)} test sentences from config")

    assert test_sentences, "No test sentences found. Use --text to provide custom text."

    # Run inference for each combination
    total_files = len(test_sentences) * len(speaker_ids)
    successful = 0

    print(f"\nGenerating {total_files} audio files...")

    for sentence_idx, sentence in enumerate(test_sentences):
        print(f"\nSentence {sentence_idx + 1}: {sentence}")

        for speaker_id in speaker_ids:
            # Create output filename
            safe_sentence = "".join(
                c for c in sentence if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            safe_sentence = safe_sentence.replace(" ", "_")[:50]  # Limit length
            output_filename = f"sentence_{sentence_idx + 1:02d}_speaker_{speaker_id}_{safe_sentence}.wav"
            output_path = output_dir / output_filename

            if run_inference(
                model_path,
                config_path,
                speakers_path,
                sentence,
                speaker_id,
                str(output_path),
                use_gpu=use_gpu,
                device=device,
                vocoder_path=vocoder_path,
                vocoder_config_path=vocoder_config_path,
            ):
                successful += 1

    print(f"\nInference complete!")
    print(f"Generated {successful}/{total_files} audio files successfully")
    print(f"Output directory: {output_dir}")

    if successful > 0:
        print(f"\nTo play the generated audio files:")
        print(f"   ls {output_dir}/*.wav")
        print(f"   # Use your preferred audio player to listen to the files")


if __name__ == "__main__":
    main()
