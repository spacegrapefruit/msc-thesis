import os
from TTS.tts.datasets import formatters


def ljspeech_liepa2(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the LJSpeech meta data file to TTS format
    https://keithito.com/LJ-Speech-Dataset/"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.strip().split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0])
            text = cols[2]
            speaker_name = cols[3]
            items.append(
                {
                    "text": text,
                    "audio_file": wav_file,
                    "speaker_name": speaker_name,
                    "root_path": root_path,
                }
            )
    return items


def text_normalizer_liepa2(text):
    """Custom text normalizer for Liepa2 dataset."""
    return text


# Register the custom formatter
formatters.register_formatter("ljspeech_liepa2", ljspeech_liepa2)
