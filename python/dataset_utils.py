import io
from pydub import AudioSegment
from tinytag import TinyTag


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
    if filename.endswith(".wav"):
        filename = filename[:-4]
    parts = filename.split("_")
    parts = [parts[0], parts[1][0], parts[1][1], parts[2][0], parts[2][1], *parts[3:]]
    parts_standardized = [
        parsing_rules[i].get(part, part) for i, part in enumerate(parts)
    ]
    return parts_standardized


def read_audio_segment(audio_data: bytes) -> AudioSegment:
    """
    Reads raw audio data and returns a pydub AudioSegment.
    Args:
        audio_data (bytes): Raw audio data in bytes.

    Returns:
        AudioSegment: Pydub AudioSegment object.
    """
    return AudioSegment.from_raw(
        io.BytesIO(audio_data), sample_width=2, frame_rate=16000, channels=1
    )


def save_audio_file(row, output_wav_path, normalize_text_fn):
    """
    Worker function to save a single audio file from Liepa-2 dataset.

    Args:
        row: A pandas Series representing a row from the DataFrame
        output_wav_path: Path to the output WAV directory
        normalize_text_fn: Function to normalize text

    Returns:
        str or None: Metadata line if successful, None if failed
    """
    # extract audio data and speaker_id from the row
    audio = row["audio"]
    speaker_id = row["speaker_id"]

    # create a unique filename
    wav_path = output_wav_path / row["file_name"]

    # resample to 22050 Hz
    audio = audio.set_frame_rate(22050).set_channels(1)
    audio.export(wav_path, format="wav")

    # get the transcript
    transcript = row["sentence"].replace("\n", " ").strip()
    normalized_transcript = normalize_text_fn(transcript)

    # return metadata line in a multi-speaker format
    # format: filename|original_transcript|normalized_transcript|speaker_id
    return f"{row['file_name']}|{transcript}|{normalized_transcript}|{speaker_id}"
