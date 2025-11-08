import argparse
import multiprocessing as mp
import pandas as pd
import re
import io
from functools import partial
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm


parsing_rules = [
    {"L": "lossy", "R": "raw"},
    {"R": "read", "S": "spontaneous"},
    {"A": "audiobook", "D": "dictaphone", "P": "phone", "R": "radio", "S": "studio", "T": "TV"},
    {"F": "female", "M": "male"},
    {"1": "0-12", "2": "13-17", "3": "18-25", "4": "26-60", "5": "60+"},
    {},
    {},
    {},
]

def parse_filename(filename):
    filename = filename[:-4]
    parts = filename.split("_")
    parts = [parts[0], parts[1][0], parts[1][1], parts[2][0], parts[2][1], *parts[3:]]
    parts_standardized = [parsing_rules[i].get(part, part) for i, part in enumerate(parts)]
    return parts_standardized


def normalize_text(text):
    """
    Normalize text for TTS training.

    Args:
        text (str): Raw text to normalize

    Returns:
        str: Normalized text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation except apostrophes (important for Lithuanian)
    # Keep basic punctuation that affects pronunciation
    text = re.sub(r"[–-]", "-", text)  # Replace dashes with space
    text = re.sub(r"[^\w\s.,\-]", "", text)

    # Remove extra whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()

    # FIXME: temporary replacements for Lithuanian-specific characters
    text = text.replace("x", "ks").replace("w", "v").replace("q", "kv")

    return text


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
        # Extract audio data and speaker_id from the row
        audio_data = row["audio"]
        audio_bytes = audio_data["bytes"]
        speaker_id = row["speaker_id"]

        # Create a unique filename based on the new DataFrame index
        wav_filename = f"{row['path']}.wav"
        wav_path = output_wav_path / wav_filename

        # Convert audio bytes to WAV using pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")

        # Get the transcript (sentence)
        transcript = row["sentence"].replace("\n", " ").strip()
        normalized_transcript = normalize_text(transcript)

        # Return metadata line in a multi-speaker format
        # format: filename|original_transcript|normalized_transcript|speaker_id
        return f"{wav_filename.replace('.wav', '')}|{transcript}|{normalized_transcript}|{speaker_id}"
    except Exception as e:
        print(f"Could not process audio at index {index}. Error: {e}")
        return None


def process_liepa2(
    liepa_path: Path,
    output_path: Path,
    n_processes: int = None,
    max_files: int = None,
    n_speakers: int = 1,
):
    """
    Prepares the Liepa-2 dataset for multi-speaker TTS training.

    Args:
        liepa_path (Path): Path to the Liepa-2 directory containing parquet files.
        output_path (Path): Path to save the processed dataset.
        n_processes (int, optional): Number of parallel processes to use. If None, uses CPU count.
        max_files (int, optional): Maximum number of files to process for testing. If None, processes all.
        n_speakers (int, optional): Number of top speakers to process. Defaults to 1.
    """
    print("Starting Liepa-2 preprocessing for multiple speakers...")

    # 1. Define paths
    output_wav_path = output_path / "wavs"

    # Create output directories
    output_path.mkdir(exist_ok=True)
    output_wav_path.mkdir(exist_ok=True)

    # 2. Load all parquet files
    print(f"Loading parquet files from: {liepa_path}")

    # Get all train parquet files
    train_files = sorted(liepa_path.glob("train-*.parquet"))
    print(f"Found {len(train_files)} training parquet files")

    # Load and concatenate all training data
    dfs = []
    for file_path in tqdm(train_files, desc="Loading parquet files"):
        df = pd.read_parquet(file_path)
        # Filter for Lithuanian language only
        df = df[df["language"] == "lt"]
        dfs.append(df)

    # Combine all dataframes
    full_df = pd.concat(dfs, ignore_index=True)

    # Extract speaker IDs
    full_df["path"] = full_df["audio"].apply(lambda x: x["path"])
    full_df[
        ["lossiness", "speech_type", "source_type", "speaker_gender", "speaker_age", "speaker_id", "recording_id", "sentence_id"]
    ] = full_df.path.apply(parse_filename).tolist()

    # Filter for read speech and specific age groups
    full_df = full_df[
        (full_df["speech_type"] == "read") &
        (full_df["speaker_age"].isin(["18-25", "26-60", "60+"]))
    ]
    print(f"Total samples loaded: {len(full_df)}")
    print(f"Unique speakers found: {full_df['speaker_id'].nunique()}")

    # FIXME: Select top N speakers based on the number of samples
    speaker_ids = full_df[["speaker_gender", "speaker_id"]].value_counts().groupby("speaker_gender").head(10).reset_index()["speaker_id"]

    # Filter the DataFrame to include only the selected speakers
    full_df = full_df[full_df["speaker_id"].isin(speaker_ids)].reset_index(drop=True)

    print(f"Selected speakers: {speaker_ids.tolist()}")
    print(f"Total samples from selected speakers: {len(full_df)}")

    # TODO: Remove - for testing only
    full_df["prefix"] = full_df["path"].str.slice(0, 13)
    print("Prefix counts:")
    print(full_df["prefix"].value_counts())

    # Limit dataset size if specified (for testing)
    if max_files:
        full_df = full_df.head(max_files)
        print(f"Limited to {len(full_df)} samples for processing")

    # 3. Set up multiprocessing
    if n_processes is None:
        n_processes = mp.cpu_count()
    print(f"Using {n_processes} processes for audio conversion")

    # 4. Process audio files
    print("Converting audio files...")
    process_func = partial(process_audio_file, output_wav_path=output_wav_path)

    with mp.Pool(n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_func, full_df.iterrows()),
                total=len(full_df),
                desc="Processing audio",
            )
        )

    # 5. Filter successful results and create metadata
    metadata = [result for result in results if result is not None]
    print(f"Successfully processed {len(metadata)} files out of {len(full_df)}")

    # 6. Save metadata file
    metadata_file_path = output_path / "metadata.csv"
    print(f"Writing metadata to: {metadata_file_path}")
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        # The metadata format now includes speaker_id at the end of each line
        for line in metadata:
            f.write(f"{line}\n")

    print(f"\nPreprocessing complete!")
    print(f"   - Processed {len(metadata)} audio files.")
    print(f"   - WAV files saved in: {output_wav_path}")
    print(f"   - Metadata file saved as: {metadata_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Liepa-2 data for multi-speaker TTS training."
    )
    parser.add_argument(
        "--liepa_path",
        type=str,
        required=True,
        help="Path to the root of the Liepa-2 directory containing parquet files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="tts_dataset_liepa2_multispeaker",
        help="Path to the directory to save the formatted dataset.",
    )
    parser.add_argument(
        "--n_speakers",
        type=int,
        default=1,
        help="Number of top speakers to process based on sample count. Default is 1.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional: Maximum number of files to process (for testing).",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=None,
        help="Number of parallel processes to use for audio processing. Default is CPU count.",
    )

    args = parser.parse_args()

    liepa_path = Path(args.liepa_path)
    output_path = Path(args.output_path)

    process_liepa2(
        liepa_path,
        output_path,
        n_processes=args.n_processes,
        max_files=args.max_files,
        n_speakers=args.n_speakers,
    )
