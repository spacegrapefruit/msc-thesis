import argparse
import json
import multiprocessing as mp
import pandas as pd
import io
from functools import partial
from pathlib import Path
from pydub import AudioSegment
from tinytag import TinyTag
from tqdm import tqdm
from text_utils import load_accented_words, normalize_text


parsing_rules = [
    {"L": "lossy", "R": "raw"},
    {"R": "read", "S": "spontaneous"},
    {
        "A": "audiobook",
        "D": "dictaphone",
        "P": "phone",
        "R": "radio",
        "S": "studio",
        "T": "TV",
    },
    {"F": "female", "M": "male"},
    {"1": "0-12", "2": "13-17", "3": "18-25", "4": "26-60", "5": "60+"},
    {},
    {},
    {},
]


def get_duration(row):
    tag = TinyTag.get(file_obj=io.BytesIO(row["audio"]["bytes"]))
    return tag.duration


def parse_filename(filename):
    filename = filename[:-4]
    parts = filename.split("_")
    parts = [parts[0], parts[1][0], parts[1][1], parts[2][0], parts[2][1], *parts[3:]]
    parts_standardized = [
        parsing_rules[i].get(part, part) for i, part in enumerate(parts)
    ]
    return parts_standardized


def parse_and_filter_df(df):
    df["path"] = df["audio"].apply(lambda x: x["path"])
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
    ] = df.path.apply(parse_filename).tolist()

    return df[
        (df["speech_type"] == "read")
        & (df["speaker_age"].isin(["18-25", "26-60", "60+"]))
    ]


def process_audio_file(row_data, output_wav_path):
    """
    Worker function to process a single audio file from Liepa-2 dataset.

    Args:
        row_data: Tuple of (index, row) from pandas DataFrame
        output_wav_path: Path to the output WAV directory

    Returns:
        str or None: Metadata line if successful, None if failed
    """
    index, row = row_data
    try:
        # extract audio data and speaker_id from the row
        audio_data = row["audio"]
        audio_bytes = audio_data["bytes"]
        speaker_id = row["speaker_id"]

        # create a unique filename
        wav_filename = f"{row['path']}.wav"
        wav_path = output_wav_path / wav_filename

        # convert audio bytes to WAV
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        # resample to 22050 Hz
        audio = audio.set_frame_rate(22050).set_channels(1)
        audio.export(wav_path, format="wav")

        # get the transcript
        transcript = row["sentence"].replace("\n", " ").strip()
        normalized_transcript = normalize_text(transcript)

        # return metadata line in a multi-speaker format
        # format: filename|original_transcript|normalized_transcript|speaker_id
        return f"{wav_filename.replace('.wav', '')}|{transcript}|{normalized_transcript}|{speaker_id}"
    except Exception as e:
        print(f"Could not process audio at index {index}. Error: {e}")
        return None


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


def process_liepa2(
    input_path: Path,
    output_path: Path,
    n_speakers_per_gender: int = 10,
):
    """
    Prepares the Liepa-2 dataset for multi-speaker TTS training.

    Args:
        input_path (Path): Path to the Liepa-2 directory containing parquet files.
        output_path (Path): Path to save the processed dataset.
        n_speakers_per_gender (int, optional): Number of top speakers per gender to process. Defaults to 10.
    """
    print("Starting Liepa-2 preprocessing for multiple speakers...")

    output_wav_path = output_path / "wavs"

    # create output directories
    output_path.mkdir(exist_ok=True)
    output_wav_path.mkdir(exist_ok=True)

    # load all parquet files
    print(f"Loading parquet files from: {input_path}")

    # Get all train parquet files
    train_files = sorted(input_path.glob("train-*.parquet"))
    print(f"Found {len(train_files)} training parquet files")

    print("First pass: determining top speakers...")
    speaker_stats = []
    for file_path in tqdm(train_files, desc="Analyzing speakers"):
        df = pd.read_parquet(file_path, columns=["audio"])
        df["duration"] = df.apply(get_duration, axis=1)
        df = parse_and_filter_df(df)

        stats = (
            df.groupby(["speaker_gender", "speaker_id"])["duration"]
            .sum()
            .reset_index(name="total_duration")
        )
        speaker_stats.append(stats)

    # calculate total count and duration per speaker
    speaker_counts_df = (
        pd.concat(speaker_stats, ignore_index=True)
        .groupby(["speaker_gender", "speaker_id"])
        .sum()
        .reset_index()
        .sort_values(by=["total_duration"], ascending=False)
    )

    selected_speakers_df = speaker_counts_df.groupby("speaker_gender").head(
        n_speakers_per_gender
    )
    selected_speakers = set(selected_speakers_df["speaker_id"].to_list())
    print(f"Selected speakers: {sorted(selected_speakers)}")

    print("Second pass: loading data for selected speakers...")
    dfs = []
    for file_path in tqdm(train_files, desc="Loading filtered data"):
        df = pd.read_parquet(file_path)
        df = parse_and_filter_df(df)
        df = df[df["speaker_id"].isin(selected_speakers)]
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df["duration"] = full_df.apply(get_duration, axis=1)

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

    n_processes = mp.cpu_count()
    print(f"Using {n_processes} processes for audio conversion")

    # process audio files
    print("Converting audio files...")
    process_func = partial(process_audio_file, output_wav_path=output_wav_path)

    with mp.Pool(n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_func, sample_df.iterrows()),
                total=len(sample_df),
                desc="Processing audio",
            )
        )

    # filter successful results
    metadata = [result for result in results if result is not None]
    print(f"Successfully processed {len(metadata)} files out of {len(sample_df)}")

    # save metadata
    metadata_file_path = output_path / "metadata.csv"
    print(f"Writing metadata to: {metadata_file_path}")
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        # metadata now includes speaker_id
        for line in metadata:
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
    print(f"   - Processed {len(metadata)} audio files.")
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
        required=True,
        help="Path to the root of the Liepa-2 directory containing parquet files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/tts_dataset_liepa2",
        help="Path to the directory to save the formatted dataset.",
    )
    parser.add_argument(
        "--n_speakers_per_gender",
        type=int,
        default=10,
        help="Number of top speakers to process based on sample count. Default is 20.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    load_accented_words(input_path / "final_accented_words.csv")

    process_liepa2(
        input_path,
        output_path,
        n_speakers_per_gender=args.n_speakers_per_gender,
    )
