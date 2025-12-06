import argparse
import logging
import os
import sys
from dataclasses import dataclass, field

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.managers import save_file
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.generic_utils import ConsoleFormatter, setup_logger
from tqdm import tqdm

import dataset_formatters


@dataclass
class ComputeEmbeddingsArgs:
    dataset_path: str = field(
        default=None, metadata={"help": "Path to the dataset directory."}
    )
    output_path: str = field(
        default=None, metadata={"help": "Path for output speakers.pth file."}
    )
    model_path: str = field(
        default="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar",
        metadata={"help": "Path to speaker encoder model. Defaults to released model."},
    )
    config_path: str = field(
        default="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json",
        metadata={
            "help": "Path to speaker encoder config. Defaults to released config."
        },
    )


def compute_embeddings(args: ComputeEmbeddingsArgs):
    """Compute speaker embeddings for the multispeaker dataset."""

    # Setup logging
    setup_logger(
        "TTS", level=logging.INFO, stream=sys.stdout, formatter=ConsoleFormatter()
    )

    # Configure dataset
    c_dataset = BaseDatasetConfig()
    c_dataset.formatter = "ljspeech_liepa2"
    c_dataset.dataset_name = "liepa2"
    c_dataset.path = args.dataset_path
    c_dataset.meta_file_train = "metadata.csv"

    # Load samples
    print(f"Loading dataset from: {args.dataset_path}")
    meta_data_train, meta_data_eval = load_tts_samples([c_dataset], eval_split=False)
    samples = meta_data_train

    print(
        f"Found {len(samples)} samples from {len(set(s['speaker_name'] for s in samples))} speakers"
    )

    encoder_manager = SpeakerManager(
        encoder_model_path=args.model_path,
        encoder_config_path=args.config_path,
        use_cuda=True,
    )

    # Set the sample rate for audio processing
    encoder_manager.encoder_config.audio.sample_rate = 22050
    encoder_manager.encoder_config.audio.win_length = 550
    encoder_manager.encoder_config.audio.hop_length = 220

    class_name_key = encoder_manager.encoder_config.class_name_key

    # Compute speaker embeddings
    speaker_mapping = {}

    print("Computing speaker embeddings...")
    for fields in tqdm(samples):
        class_name = fields[class_name_key]
        audio_file = fields["audio_file"]
        embedding_key = fields["audio_unique_name"]

        # Extract the embedding
        embedd = encoder_manager.compute_embedding_from_clip(audio_file)

        # Create speaker mapping entry
        speaker_mapping[embedding_key] = {"name": class_name, "embedding": embedd}

    assert len(speaker_mapping) == len(samples), (
        "Speaker mapping size does not match number of samples! "
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_file(speaker_mapping, args.output_path)
    print(f"Speaker embeddings saved at: {args.output_path}")
    print(f"Computed embeddings for {len(speaker_mapping)} audio clips")


def main():
    parser = argparse.ArgumentParser(
        description="Compute speaker embeddings for multispeaker Liepa-2 dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the multispeaker dataset directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path for output speakers.pth file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar",
        help="Path to speaker encoder model. Defaults to released model.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json",
        help="Path to speaker encoder config. Defaults to released config.",
    )

    args = parser.parse_args()

    # Create args object
    compute_args = ComputeEmbeddingsArgs(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        model_path=args.model_path,
        config_path=args.config_path,
    )

    compute_embeddings(compute_args)


if __name__ == "__main__":
    main()
