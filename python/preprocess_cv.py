import argparse
import multiprocessing as mp
import pandas as pd
import re
from functools import partial
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm


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
    text = re.sub(r"[–-]", " ", text)  # Replace dashes with space
    text = re.sub(r"[^\w\s.]", "", text)

    # Remove extra whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()

    # FIXME: temporary replacements for Lithuanian-specific characters
    text = text.replace("x", "ks").replace("w", "v").replace("q", "kv")

    return text


def process_audio_file(row_data, clips_path, output_wav_path):
    """
    Worker function to process a single audio file.

    Args:
        row_data: Tuple of (index, row) from pandas DataFrame
        clips_path: Path to the clips directory
        output_wav_path: Path to the output WAV directory

    Returns:
        str or None: Metadata line if successful, None if failed
    """
    index, row = row_data
    try:
        mp3_path = clips_path / row["path"]
        wav_filename = Path(row["path"]).with_suffix(".wav")
        wav_path = output_wav_path / wav_filename

        # Convert MP3 to WAV using pydub
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(22050).set_channels(1)
        audio.export(wav_path, format="wav")

        # Get the transcript (sentence)
        transcript = row["sentence"]
        normalized_transcript = normalize_text(transcript)

        # Return metadata line in LJSpeech format (filename|original_transcript|normalized_transcript)
        return f"{wav_filename.stem}|{transcript}|{normalized_transcript}|"
    except Exception as e:
        print(f"Could not process file {mp3_path}. Error: {e}")
        return None


def process_common_voice(
    cv_path: Path,
    output_path: Path,
    target_speaker_id: str = None,
    n_processes: int = None,
):
    """
    Prepares the Common Voice dataset for single-speaker Tacotron 2 training.

    Args:
        cv_path (Path): Path to the extracted Common Voice directory (e.g., './cv-corpus-11.0-2022-09-21/lt/').
        output_path (Path): Path to save the processed dataset.
        target_speaker_id (str, optional): Manually specify a speaker ID.
                                         If None, the speaker with the most clips is chosen automatically.
        n_processes (int, optional): Number of parallel processes to use. If None, uses CPU count.
    """
    print("Starting Common Voice preprocessing...")

    # 1. Define paths
    tsv_path = cv_path / "validated.tsv"
    clips_path = cv_path / "clips"
    output_wav_path = output_path / "wavs"

    # Create output directories
    output_path.mkdir(exist_ok=True)
    output_wav_path.mkdir(exist_ok=True)

    # 2. Load and analyze the dataset metadata
    print(f"Loading metadata from: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t", usecols=["client_id", "path", "sentence"])

    if not target_speaker_id:
        print("Finding the best speaker (with the most clips)...")
        speaker_counts = df["client_id"].value_counts()
        target_speaker_id = speaker_counts.index[0]
        print(
            f"Best speaker found: {target_speaker_id} with {speaker_counts.iloc[0]} clips."
        )
    else:
        print(f"Using manually specified speaker: {target_speaker_id}")

    # 3. Filter the dataframe for the target speaker
    speaker_df = df[df["client_id"] == target_speaker_id].copy()
    print(f"Found {len(speaker_df)} clips for speaker {target_speaker_id}.")

    # 4. Process audio and create metadata list
    if n_processes is None:
        n_processes = mp.cpu_count()

    print(
        f"Processing audio clips and converting to WAV (22050 Hz, Mono) using {n_processes} processes..."
    )

    # Create a partial function with fixed arguments
    worker_func = partial(
        process_audio_file, clips_path=clips_path, output_wav_path=output_wav_path
    )

    # Prepare data for parallel processing
    row_data = list(speaker_df.iterrows())

    # Process files in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(worker_func, row_data),
                total=len(row_data),
                desc="Processing audio files",
            )
        )

    # Filter out None results (failed processing)
    metadata = [result for result in results if result is not None]

    # 5. Write the metadata file
    metadata_file_path = output_path / "metadata.csv"
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        for line in metadata:
            f.write(f"{line}\n")

    print(f"\nPreprocessing complete!")
    print(f"   - Processed {len(metadata)} audio files.")
    print(f"   - WAV files saved in: {output_wav_path}")
    print(f"   - Metadata file saved as: {metadata_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Common Voice data for TTS training."
    )
    parser.add_argument(
        "--cv_path",
        type=str,
        required=True,
        help="Path to the root of the extracted CV directory for your language.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="tts_dataset",
        help="Path to the directory to save the formatted dataset.",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default=None,
        help="Optional: Manually specify the client_id of the speaker to use.",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=None,
        help="Number of parallel processes to use for audio processing. Default is CPU count.",
    )

    args = parser.parse_args()

    cv_path = Path(args.cv_path)
    output_path = Path(args.output_path)

    process_common_voice(cv_path, output_path, args.speaker_id, args.n_processes)
