import argparse
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dataset_utils import get_duration, parse_filename, save_audio_file
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


def create_testset(
    input_path: Path,
    datasets_path: Path,
    output_path: Path,
    n_phrases_per_speaker: int = 10,
):
    """
    Prepare a test set from the Liepa-2 dataset for Lithuanian TTS evaluation.
    """
    print("Starting Liepa-2 test set preparation...")

    output_wav_path = output_path / "wavs"

    # create output directories
    output_path.mkdir(exist_ok=True)
    output_wav_path.mkdir(exist_ok=True)

    common_speakers = None
    all_trainsets = []
    for dataset_dir in datasets_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        if dataset_dir.name == "liepa2_test_set":
            continue

        metadata_file = dataset_dir / "metadata.csv"
        trainset_df = pd.read_csv(metadata_file, sep="|", header=None)
        trainset_df.columns = [
            "filename",
            "original_transcript",
            "normalized_transcript",
            "speaker_id",
        ]
        all_trainsets.append(trainset_df)

        if common_speakers is None:
            common_speakers = set(trainset_df["speaker_id"].unique())
        else:
            common_speakers &= set(trainset_df["speaker_id"].unique())

    common_speakers = sorted(common_speakers)
    print(f"Common speakers across datasets: {common_speakers}")

    # Fixed seed for reproducibility
    random.seed(201746)
    selected_speakers = random.sample(common_speakers, k=6)
    print(f"Selected speakers for test set: {sorted(selected_speakers)}")

    all_trainsets_df = pd.concat(all_trainsets, ignore_index=True)
    all_trainsets_df = all_trainsets_df.drop_duplicates(
        subset=["filename"]
    ).reset_index(drop=True)
    print(f"Total unique samples in combined trainsets: {len(all_trainsets_df)}")

    # load all parquet files
    print(f"Loading parquet files from: {input_path}")

    # Get all train parquet files
    train_files = sorted(input_path.glob("train-*.parquet"))
    print(f"Found {len(train_files)} training parquet files")

    all_dfs = []
    for file_path in tqdm(train_files, desc="Loading and filtering data"):
        df = pd.read_parquet(file_path, columns=["audio", "sentence"])
        df["duration"] = df.apply(get_duration, axis=1)
        df = parse_and_filter_df(df)
        df = df[df["speaker_id"].isin(selected_speakers)]
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    # filter out samples shorter than 3 seconds and longer than 15 seconds
    full_df = full_df[(full_df["duration"] >= 3.0) & (full_df["duration"] <= 15.0)]
    # sentence starts with uppercase letter
    full_df["is_upper"] = full_df["sentence"].apply(lambda x: x[0].isupper())
    full_df.sort_values(
        by=["is_upper", "speaker_id", "duration"],
        ascending=[False, True, True],
        inplace=True,
    )
    print(f"Total samples after filtering: {len(full_df)}")

    unseen_samples = []
    for speaker_id in sorted(selected_speakers):
        speaker_df = full_df[full_df["speaker_id"] == speaker_id]
        speaker_trainset_filenames = set(
            all_trainsets_df[all_trainsets_df["speaker_id"] == speaker_id]["filename"]
        )

        speaker_unseen_df = speaker_df[
            ~speaker_df["path"].isin(speaker_trainset_filenames)
        ]

        samples_df = speaker_unseen_df.head(n_phrases_per_speaker)
        unseen_samples.append(samples_df)
    test_set_df = pd.concat(unseen_samples, ignore_index=True)
    print(f"Total samples in the test set: {len(test_set_df)}")

    # show speaker distribution
    speaker_distribution = (
        test_set_df.groupby(["speaker_gender", "speaker_id"])
        .size()
        .reset_index(name="count")
        .sort_values(by=["speaker_id"])
    )
    print("\nSpeaker distribution:")
    for _, row in speaker_distribution.iterrows():
        print(
            f"  {row['speaker_id']} ({row['speaker_gender']}): {row['count']} samples"
        )

    print("Converting audio files...")
    metadata = []
    for index, row in tqdm(
        test_set_df.iterrows(), total=len(test_set_df), desc="Processing audio"
    ):
        result = save_audio_file(
            row, output_wav_path, normalize_text_fn=normalize_text
        )
        if result:
            metadata.append(result)

    print(f"Successfully processed {len(metadata)} files out of {len(test_set_df)}")

    metadata_file_path = output_path / "metadata.csv"
    print(f"Writing metadata to: {metadata_file_path}")
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        for line in metadata:
            f.write(f"{line}\n")

    print(f"\nPreprocessing complete!")
    print(f"   - Processed {len(metadata)} audio files.")
    print(f"   - Number of unique speakers: {len(selected_speakers)}")
    print(f"   - WAV files saved in: {output_wav_path}")
    print(f"   - Metadata file saved as: {metadata_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a test-set for Lithuanian TTS evaluation from Liepa-2 dataset."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the root of the Liepa-2 directory containing parquet files.",
    )
    parser.add_argument(
        "--datasets_path",
        type=str,
        required=True,
        help="Path to the root of the prepared datasets directory.",
    )
    parser.add_argument(
        "--n_phrases_per_speaker",
        type=int,
        default=10,
        help="Number of phrases to sample per speaker for the test set.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    datasets_path = Path(args.datasets_path)
    output_path = datasets_path / "liepa2_test_set"

    load_accented_words(input_path / "final_accented_words.csv")

    create_testset(
        input_path=input_path,
        datasets_path=datasets_path,
        output_path=output_path,
        n_phrases_per_speaker=args.n_phrases_per_speaker,
    )
