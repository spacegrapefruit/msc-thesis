import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import librosa
import numpy as np
import pandas as pd
import pyworld as pw
from speechmos import dnsmos
from fastdtw import fastdtw
from pesq import pesq
from scipy.spatial.distance import euclidean
from tqdm import tqdm

warnings.filterwarnings("ignore")


def get_hop_length(sr, frame_period_ms=5.0):
    return int(sr * (frame_period_ms / 1000))


def extract_features(audio, sr, frame_period_ms=5.0, n_mcep=24):
    audio = audio.astype(np.float64)
    f0, timeaxis = pw.dio(audio, sr, frame_period=frame_period_ms)
    f0 = pw.stonemask(audio, f0, timeaxis, sr)
    sp = pw.cheaptrick(audio, f0, timeaxis, sr)
    mcep = pw.code_spectral_envelope(sp, sr, n_mcep)
    min_len = min(len(f0), len(mcep))
    f0 = f0[:min_len]
    mcep = mcep[:min_len]
    return mcep, f0


def compute_metrics_with_dtw(mfcc_ref, mfcc_synth, f0_ref, f0_synth):
    m_ref = mfcc_ref[:, 1:]
    m_synth = mfcc_synth[:, 1:]
    dist, path = fastdtw(m_ref, m_synth, dist=euclidean)
    path_ref = [p[0] for p in path]
    path_synth = [p[1] for p in path]
    diff_squared = (m_ref[path_ref] - m_synth[path_synth]) ** 2
    mcd_frames = np.sqrt(2 * np.sum(diff_squared, axis=1))
    K = 10.0 / np.log(10.0)
    mcd = K * np.mean(mcd_frames)
    f0_ref_aligned = f0_ref[path_ref]
    f0_synth_aligned = f0_synth[path_synth]
    voiced_mask = (f0_ref_aligned > 0) & (f0_synth_aligned > 0)
    if np.sum(voiced_mask) == 0:
        f0_rmse = np.nan
    else:
        f0_diff = f0_ref_aligned[voiced_mask] - f0_synth_aligned[voiced_mask]
        f0_rmse = np.sqrt(np.mean(f0_diff**2))
    return mcd, f0_rmse


def load_audio(audio_path, sr=22050):
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    return audio


def parse_filename(filename):
    if not filename.endswith(".wav"):
        return None
    base = filename[:-4]
    parts = base.split("-")
    if len(parts) < 4:
        return None
    if parts[0] == "Tacotron2" and len(parts) > 4:
        return {
            "model_name": f"{parts[0]}-{parts[1]}",
            "model_version": parts[2],
            "phrase_id": parts[3],
            "speaker_id": parts[4],
        }
    else:
        return {
            "model_name": parts[0],
            "model_version": parts[1],
            "phrase_id": parts[2],
            "speaker_id": parts[3],
        }


def evaluate_sample(synth_path, gt_path, sr=22050):
    audio_synth = load_audio(synth_path, sr)
    audio_gt = load_audio(gt_path, sr)
    mfcc_synth, f0_synth = extract_features(audio_synth, sr)
    mfcc_gt, f0_gt = extract_features(audio_gt, sr)
    mcd, f0_rmse = compute_metrics_with_dtw(mfcc_gt, mfcc_synth, f0_gt, f0_synth)

    # Compute PESQ (requires 16kHz or 8kHz)
    if sr == 16000:
        pesq_mode = "wb"  # wideband
    elif sr == 8000:
        pesq_mode = "nb"  # narrowband
    else:
        # Resample to 16kHz for PESQ
        audio_gt_16k = librosa.resample(audio_gt, orig_sr=sr, target_sr=16000)
        audio_synth_16k = librosa.resample(audio_synth, orig_sr=sr, target_sr=16000)
        pesq_mode = "wb"
        audio_gt = audio_gt_16k
        audio_synth = audio_synth_16k

    pesq_score = pesq(
        16000 if pesq_mode == "wb" else 8000, audio_gt, audio_synth, pesq_mode
    )

    # normalize audio for DNSMOS
    audio_synth_norm = audio_synth / (np.max(np.abs(audio_synth)) + 1e-7)
    utmos_score = dnsmos.run(audio_synth_norm, sr=16000)["ovrl_mos"]

    return {
        "mcd": mcd,
        "f0_rmse": f0_rmse,
        "pesq": pesq_score,
        "utmos": utmos_score,
    }


def evaluate_all_samples(samples_dir: Path, output_file: Path, sr=22050, num_workers=8):
    gt_dir = samples_dir / "ground_truth"
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    # Find all model directories
    model_dirs = [
        d for d in samples_dir.iterdir() if d.is_dir() and d.name != "ground_truth"
    ]

    if not model_dirs:
        raise ValueError("No model directories found")

    print(f"Found {len(model_dirs)} model directories")
    print(f"Using {num_workers} workers for parallel processing")

    # Build ground truth mapping: phrase_id -> ground truth file
    gt_mapping = {}
    for gt_file in gt_dir.glob("*.wav"):
        parsed = parse_filename(gt_file.name)
        if parsed:
            # GT files are named like: GT-000000-p001-spk123.wav
            phrase_id = parsed["phrase_id"]
            speaker_id = parsed["speaker_id"]
            gt_mapping[(phrase_id, speaker_id)] = gt_file

    print(f"Found {len(gt_mapping)} ground truth files")

    # Collect all evaluation tasks
    tasks = []
    for model_dir in model_dirs:
        for synth_file in model_dir.glob("*.wav"):
            parsed = parse_filename(synth_file.name)
            if not parsed:
                continue

            phrase_id = parsed["phrase_id"]
            speaker_id = parsed["speaker_id"]
            model_name = parsed["model_name"]
            model_version = parsed["model_version"]

            # Find corresponding ground truth
            gt_key = (phrase_id, speaker_id)
            if gt_key not in gt_mapping:
                print(f"Warning: No ground truth found for {synth_file.name}")
                continue

            gt_file = gt_mapping[gt_key]

            tasks.append(
                {
                    "synth_path": synth_file,
                    "gt_path": gt_file,
                    "model_name": model_name,
                    "model_version": model_version,
                    "phrase_id": phrase_id,
                    "speaker_id": speaker_id,
                    "sr": sr,
                }
            )

    print(f"Prepared {len(tasks)} evaluation tasks")

    # Evaluate in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                evaluate_sample, task["synth_path"], task["gt_path"], task["sr"]
            ): task
            for task in tasks
        }

        # Process results with progress bar
        with tqdm(total=len(tasks), desc="Evaluating samples") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                result = future.result()

                results.append(
                    {
                        "model_name": task["model_name"],
                        "model_version": task["model_version"],
                        "phrase_id": task["phrase_id"],
                        "speaker_id": task["speaker_id"],
                        "mcd": result["mcd"],
                        "f0_rmse": result["f0_rmse"],
                        "pesq": result["pesq"],
                        "utmos": result["utmos"],
                    }
                )

                pbar.update(1)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save detailed results
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    # Compute and display statistics per model
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print(results_df.sort_values(by=["utmos"], ascending=False))

    model_stats = (
        results_df.groupby(["model_name", "model_version"])
        .agg(
            {
                "mcd": ["mean", "std", "count"],
                "f0_rmse": ["mean", "std"],
                "pesq": ["mean", "std"],
                "utmos": ["mean", "std"],
            }
        )
        .round(4)
    )

    print("\nPer-Model Statistics:")
    print(model_stats)

    print("\n" + "-" * 80)
    print(
        f"Overall MCD: {results_df['mcd'].mean():.4f} ± {results_df['mcd'].std():.4f}"
    )
    print(
        f"Overall F0 RMSE: {results_df['f0_rmse'].mean():.4f} ± {results_df['f0_rmse'].std():.4f}"
    )
    print(
        f"Overall PESQ: {results_df['pesq'].mean():.4f} ± {results_df['pesq'].std():.4f}"
    )
    print(
        f"Overall UTMOS: {results_df['utmos'].mean():.4f} ± {results_df['utmos'].std():.4f}"
    )
    print(f"Total evaluated: {len(results_df)} samples")
    print("-" * 80)

    # Save summary statistics
    summary_file = output_file.parent / f"{output_file.stem}_summary.csv"
    model_stats.to_csv(summary_file)
    print(f"\nSummary statistics saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MCD, F0 RMSE, PESQ, and UTMOS"
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        default="results/test_samples",
        help="Directory containing synthesized samples and ground truth",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/objective_metrics.csv",
        help="Output CSV file for detailed results",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sample rate for audio processing (default: 22050)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )

    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    output_file = Path(args.output_file)

    # Create output directory if needed
    output_file.parent.mkdir(exist_ok=True, parents=True)

    evaluate_all_samples(
        samples_dir=samples_dir,
        output_file=output_file,
        sr=args.sr,
        num_workers=args.num_workers,
    )
