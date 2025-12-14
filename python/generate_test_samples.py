import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from text_utils import load_accented_words


EXCLUDED_MODELS = [
    "Glow-December-13-2025_11+03AM-171616c",
    "Tacotron2-DCA-November-20-2025_09+07PM-da89dae",
    "Tacotron2-DCA-November-22-2025_01+41PM-2b683eb",
    "Tacotron2-DCA-November-23-2025_09+47AM-e6df90a",
    "Tacotron2-DCA-December-03-2025_11+35PM-9eeb05d",
]


def get_model_info(model_dir):
    """Extract model name and version from directory name."""
    dir_name = model_dir.name
    parts = dir_name.split("-")
    model_name = parts[0]

    # model version: 030spk/060spk/180spk based on num_speakers in config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
        num_speakers = config["num_speakers"]
        model_version = f"{num_speakers:03d}spk"
    return model_name, model_version


def get_model_checkpoint(model_dir):
    """Find the best model checkpoint in the training directory."""
    best_model_path = model_dir / "best_model.pth"
    config_path = model_dir / "config.json"
    speakers_path = model_dir / "speakers.pth"

    if not all([best_model_path.exists(), config_path.exists()]):
        return None, None, None

    return (
        str(best_model_path),
        str(config_path),
        str(speakers_path) if speakers_path.exists() else None,
    )


def run_tts_inference(
    model_path,
    config_path,
    speakers_path,
    text,
    speaker_id,
    output_path,
    use_gpu=True,
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
        str(output_path),
    ]

    if speakers_path:
        cmd.extend(["--speakers_file_path", speakers_path])

    if speaker_id is not None:
        cmd.extend(["--speaker_idx", speaker_id])

    if vocoder_path:
        cmd.extend(["--vocoder_path", vocoder_path])
        if vocoder_config_path:
            cmd.extend(["--vocoder_config_path", vocoder_config_path])

    if use_gpu:
        cmd.append("--use_cuda")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating {output_path}: {e.stderr}")
        return False


def generate_test_samples(
    test_set_path: Path,
    model_dirs: list[Path],
    output_dir: Path,
    use_gpu: bool = True,
    vocoder_path: str = None,
    vocoder_config_path: str = None,
):
    """Generate synthesized audio for all test sentences using specified models."""

    metadata_path = test_set_path / "metadata.csv"
    ground_truth_wavs = test_set_path / "wavs"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path, sep="|", header=None)
    metadata.columns = ["filename", "original_text", "normalized_text", "speaker_id"]

    print(f"Loaded {len(metadata)} test samples")
    print(f"Unique speakers: {sorted(metadata['speaker_id'].unique())}")

    output_dir.mkdir(exist_ok=True, parents=True)
    ground_truth_dir = output_dir / "ground_truth"
    ground_truth_dir.mkdir(exist_ok=True)

    print("\nCopying ground truth audio files...")
    for row_i, row in tqdm(
        metadata.iterrows(), total=len(metadata), desc="Ground truth"
    ):
        src_file = ground_truth_wavs / f"{row['filename']}.wav"
        if src_file.exists():
            dst_filename = f"GT-000000-p{row_i + 1:03d}-{row['speaker_id']}.wav"
            dst_file = ground_truth_dir / dst_filename
            shutil.copy2(src_file, dst_file)

    for model_dir in model_dirs:
        model_name, model_version = get_model_info(model_dir)
        model_path, config_path, speakers_path = get_model_checkpoint(model_dir)

        if model_path is None:
            print(f"Skipping {model_dir.name} - no valid checkpoint found")
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing model: {model_name}-{model_version}")
        print(f"Model directory: {model_dir.name}")
        print(f"{'=' * 80}")

        model_output_dir = output_dir / f"{model_name}-{model_version}"
        model_output_dir.mkdir(exist_ok=True)

        successful = 0
        total = len(metadata)

        for row_i, row in tqdm(
            metadata.iterrows(), total=total, desc=f"{model_name}-{model_version}"
        ):
            speaker_id = row["speaker_id"]
            text = row["normalized_text"]

            output_filename = (
                f"{model_name}-{model_version}-p{row_i + 1:03d}-{speaker_id}.wav"
            )
            output_path = model_output_dir / output_filename

            if run_tts_inference(
                model_path,
                config_path,
                speakers_path,
                text,
                speaker_id,
                output_path,
                use_gpu=use_gpu,
                vocoder_path=vocoder_path,
                vocoder_config_path=vocoder_config_path,
            ):
                successful += 1

        print(f"Generated {successful}/{total} audio files successfully")

    print(f"\n{'=' * 80}")
    print("Generation complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthesized audio samples for test set evaluation"
    )
    parser.add_argument(
        "--test_set_path",
        type=str,
        default="data/processed/liepa2_test_set",
        help="Path to the test set directory",
    )
    parser.add_argument(
        "--model_dirs",
        type=str,
        nargs="+",
        help="List of model directories to use for generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/test_samples",
        help="Output directory for generated audio files",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="Use GPU for inference",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        default=os.path.expanduser("~/.local/share/tts/vocoder_models--en--vctk--hifigan_v2/model.pth"),
        help="Path to vocoder model file",
    )
    parser.add_argument(
        "--vocoder_config_path",
        type=str,
        default=os.path.expanduser("~/.local/share/tts/vocoder_models--en--vctk--hifigan_v2/config.json"),
        help="Path to vocoder config file",
    )

    args = parser.parse_args()

    input_path = Path("data/raw/liepa2")
    load_accented_words(input_path / "final_accented_words.csv")

    test_set_path = Path(args.test_set_path)
    output_dir = Path(args.output_dir)

    if args.model_dirs:
        model_dirs = [Path(d) for d in args.model_dirs]
    else:
        training_output = Path("training_output")
        model_dirs = sorted(training_output.glob("*-*-*"))
        model_dirs = [
            d for d in model_dirs if d.is_dir() and d.name not in EXCLUDED_MODELS
        ]
        print(f"Auto-detected {len(model_dirs)} model directories")

    model_dirs = [d for d in model_dirs if d.is_dir()]

    if not model_dirs:
        raise ValueError("No model directories found")

    print(f"Will process {len(model_dirs)} models:")
    for d in model_dirs:
        print(f"  - {d.name}")

    generate_test_samples(
        test_set_path=test_set_path,
        model_dirs=model_dirs,
        output_dir=output_dir,
        use_gpu=args.use_gpu,
        vocoder_path=args.vocoder_path,
        vocoder_config_path=args.vocoder_config_path,
    )
