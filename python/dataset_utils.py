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
    filename = filename[:-4]
    parts = filename.split("_")
    parts = [parts[0], parts[1][0], parts[1][1], parts[2][0], parts[2][1], *parts[3:]]
    parts_standardized = [
        parsing_rules[i].get(part, part) for i, part in enumerate(parts)
    ]
    return parts_standardized


def process_audio_file(row_data, output_wav_path, normalize_text_fn):
    """
    Worker function to process a single audio file from Liepa-2 dataset.

    Args:
        row_data: Tuple of (index, row) from pandas DataFrame
        output_wav_path: Path to the output WAV directory
        normalize_text_fn: Function to normalize text

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
        normalized_transcript = normalize_text_fn(transcript)

        # return metadata line in a multi-speaker format
        # format: filename|original_transcript|normalized_transcript|speaker_id
        return f"{wav_filename.replace('.wav', '')}|{transcript}|{normalized_transcript}|{speaker_id}"
    except Exception as e:
        print(f"Could not process audio at index {index}. Error: {e}")
        return None
