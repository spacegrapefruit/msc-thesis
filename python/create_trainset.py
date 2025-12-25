import argparse
import io
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dataset_utils import parse_filename, save_audio_file
from pydub import AudioSegment
from text_utils import load_accented_words, normalize_text


def parse_and_filter_df(df):
    df[
        [
            "lossiness",
            "speech_type",
            "source_type",
            "speaker_gender",
            "speaker_age",
            "speaker_id",
            "recording_id",
            "sentence_id",
        ]
    ] = df["file_name"].apply(parse_filename).tolist()

    return df[
        (df["speech_type"] == "read")
        & (df["speaker_age"].isin(["18-25", "26-60", "60+"]))
    ]


def sample_duration_per_speaker(df, selected_speakers, seconds_per_speaker):
    selected_df = df[df["speaker_id"].isin(selected_speakers)]
    sampled_dfs = []
    for speaker_id in selected_speakers:
        speaker_df = selected_df[selected_df["speaker_id"] == speaker_id]
        speaker_df = speaker_df.sample(
            frac=1, random_state=42
        )  # shuffle rows for randomness
        cumulative_duration = (
            speaker_df["duration"].cumsum() - speaker_df["duration"]
        )  # include one more utterance
        speaker_sampled_df = speaker_df[cumulative_duration <= seconds_per_speaker]
        sampled_dfs.append(speaker_sampled_df)
    return pd.concat(sampled_dfs, ignore_index=True)


def create_trainset(
    input_path: Path,
    output_path: Path,
    n_speakers_per_gender: int = 15,
):
    """
    Prepares the Liepa-2 dataset for multi-speaker TTS training.

    Args:
        input_path (Path): Path to the Liepa-2 directory containing sliced audio data.
        output_path (Path): Path to save the processed dataset.
        n_speakers_per_gender (int, optional): Number of top speakers per gender to process. Defaults to 15.
    """
    print("Starting Liepa-2 preprocessing for multiple speakers...")

    output_wav_path = output_path / "wavs"

    # create output directories
    output_path.mkdir(exist_ok=True)
    output_wav_path.mkdir(exist_ok=True)

    # load all parquet files
    print(f"Loading sliced audio data from: {input_path}")

    def read_audio_segment(audio_data):
        return AudioSegment.from_raw(
            io.BytesIO(audio_data), sample_width=2, frame_rate=16000, channels=1
        )

    full_df = pd.read_parquet(input_path / "sliced_dataset.parquet")
    full_df["audio"] = full_df["audio"].apply(read_audio_segment)
    full_df["duration"] = full_df["audio"].apply(lambda x: x.duration_seconds)
    print(f"Total samples in metadata: {len(full_df)}")

    print("Determining top speakers...")

    full_df = parse_and_filter_df(full_df)

    speaker_stats_df = (
        full_df.groupby(["speaker_gender", "speaker_id"])["duration"]
        .sum()
        .reset_index(name="total_duration")
    )

    # calculate total count and duration per speaker
    speaker_counts_df = (
        speaker_stats_df.groupby(["speaker_gender", "speaker_id"])
        .sum()
        .reset_index()
        .sort_values(by=["total_duration"], ascending=False)
    )

    selected_speakers_df = speaker_counts_df.groupby("speaker_gender").head(
        n_speakers_per_gender
    )
    selected_speakers = set(selected_speakers_df["speaker_id"].to_list())
    print(f"Selected speakers: {sorted(selected_speakers)}")

    print("Saving data for selected speakers...")
    full_df = full_df[full_df["speaker_id"].isin(selected_speakers)]

    print(f"Total samples loaded: {len(full_df)}")
    print(f"Unique speakers found: {full_df['speaker_id'].nunique()}")

    longer_than_15_sec = full_df["duration"] > 15.0
    full_df = full_df[~longer_than_15_sec]
    print(
        f"Samples longer than 15 sec: {longer_than_15_sec.sum()}, remaining samples: {len(full_df)}"
    )

    duration_per_speaker = 22.5 * 3600 / (n_speakers_per_gender * 2)
    sample_df = sample_duration_per_speaker(
        full_df, sorted(selected_speakers), duration_per_speaker
    )
    print(f"Total samples from selected speakers: {len(sample_df)}")

    # show speaker distribution
    speaker_distribution = (
        sample_df.groupby(["speaker_gender", "speaker_id"])
        .agg(total_utterances=("sentence", "count"), total_duration=("duration", "sum"))
        .reset_index()
        .sort_values(by=["speaker_id"], ascending=False)
    )
    print("\nSpeaker distribution:")
    for _, row in speaker_distribution.iterrows():
        print(
            f"  {row['speaker_id']} ({row['speaker_gender']}): {row['total_utterances']} samples, {row['total_duration']:.2f} sec"
        )

    results = [
        save_audio_file(
            row,
            output_wav_path=output_wav_path,
            normalize_text_fn=normalize_text,
        )
        for _, row in tqdm(
            sample_df.iterrows(), total=len(sample_df), desc="Saving audio files"
        )
    ]
    print(f"Successfully saved {len(results)} audio files.")

    # save metadata
    metadata_file_path = output_path / "metadata.csv"
    print(f"Writing metadata to: {metadata_file_path}")
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        # metadata now includes speaker_id
        for line in results:
            f.write(f"{line}\n")

    # create speakers.json for training
    speakers_info = {}
    unique_speakers = sample_df["speaker_id"].unique()
    for i, speaker_id in enumerate(sorted(unique_speakers)):
        speakers_info[speaker_id] = i

    speakers_file_path = output_path / "speakers.json"
    print(f"Writing speakers file to: {speakers_file_path}")
    with open(speakers_file_path, "w", encoding="utf-8") as f:
        json.dump(speakers_info, f, ensure_ascii=False, indent=2)

    print(f"\nPreprocessing complete!")
    print(f"   - Processed {len(results)} audio files.")
    print(f"   - Number of unique speakers: {len(speakers_info)}")
    print(f"   - WAV files saved in: {output_wav_path}")
    print(f"   - Metadata file saved as: {metadata_file_path}")
    print(f"   - Speakers file saved as: {speakers_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Liepa-2 data for multi-speaker TTS training."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed",
        help="Path to the root directory with sliced Liepa-2 data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/datasets",
        help="Path to the directory to save the formatted dataset.",
    )
    parser.add_argument(
        "--n_speakers_per_gender",
        type=int,
        default=15,
        help="Number of top speakers per gender to process based on sample count. Default is 15.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    load_accented_words(input_path / "final_accented_words.csv")

    create_trainset(
        input_path,
        output_path,
        n_speakers_per_gender=args.n_speakers_per_gender,
    )
