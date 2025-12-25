import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm


MAX_WORKERS = 32

PHRASES_TO_EXCLUDE = [
    "+BREATH+",
    "+COUGH+",
    "+LAUGH+",
    "+SMACK+",
    "+AH+",
    "+EH+",
    "+MM+",
    "+GARBAGE+",
    "+NOISE+",
]


def process_media(media_path: Path, segment_name: str, segments: list):
    audio = AudioSegment.from_file(media_path)

    # resample to 16 kHz and mono
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)
    audio = audio.set_channels(1)

    metadata = []

    for index, segment in enumerate(segments, start=1):
        start_time = segment["beg"]
        end_time = segment["end"]
        length = segment["len"]

        phrase = segment["val"].strip()
        audio_name = f"{segment_name}_{index:06}.wav"

        if phrase not in PHRASES_TO_EXCLUDE and length >= 1000:
            audio_segment = audio[start_time:end_time]
            # output_file = output_dir / audio_name
            # audio_segment.export(
            #     output_file,
            #     format="wav",
            #     codec="pcm_s16le",
            # )

            metadata.append(
                {
                    "file_name": audio_name,
                    "sentence": phrase,
                    "audio": audio_segment.raw_data,
                    "language": "lt",
                }
            )

    return metadata


def main(args: argparse.Namespace):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    speakers_path = input_dir / "included_speakers.yml"
    with open(speakers_path) as file:
        whitelisted_speakers = yaml.safe_load(file)

    corpus_meta_path = input_dir / "etc" / "corpus-data.json"
    with open(corpus_meta_path) as file:
        corpus_data = json.load(file)

    tasks = []
    for _, value in corpus_data.items():
        media_subpath = value["media"]["path"]
        media_path = input_dir / media_subpath
        file_stem = media_path.stem

        tiers = value.get("tiers", None)
        if tiers is None:
            segments = value["speech"]
            speaker_id = file_stem.split("_")[3]
            if speaker_id not in whitelisted_speakers:
                continue

            tasks.append((media_path, file_stem, segments))

        else:
            for tier_name in tiers:
                speaker_id = tier_name.split("_")[3]
                if speaker_id not in whitelisted_speakers:
                    continue

                segments = value["speech"][tier_name]

                if isinstance(segments, dict):
                    segments = [segments]
                tasks.append((media_path, tier_name, segments))

    print(f"{len(tasks)} tasks to process")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_media, *task) for task in tasks]

        # collect results
        all_metadata = []
        for future in tqdm(futures, total=len(futures), desc="Processing media files"):
            all_metadata.extend(future.result())

    all_metadata_df = pd.DataFrame(all_metadata)
    all_metadata_df.to_parquet(output_dir / "sliced_dataset.parquet", index=False)
    print(f"Saved {len(all_metadata_df)} segments to sliced_dataset.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Input directory",
        type=str,
        default="data/raw",
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory",
        type=str,
        default="data/processed",
    )

    args = parser.parse_args()
    main(args)
