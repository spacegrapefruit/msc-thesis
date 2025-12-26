#!/usr/bin/env python3
"""
Script to generate all figures for the MSc thesis on TTS synthesis.
This script creates publication-ready figures for LaTeX inclusion.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import seaborn as sns
from pathlib import Path
import librosa

# Configure matplotlib for LaTeX documents
plt.rcParams.update(
    {
        "text.usetex": False,  # Set to True if LaTeX is installed
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        # "figure.titlesize": 14,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

# Set the style for publication-ready figures
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Create output directory
FIGURES_DIR = Path("latex/paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def create_sampling_quantization_figure():
    """
    Create a visual representation of Analog-to-Digital conversion showing
    sampling rate and quantization levels.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Generate continuous analog signal
    t = np.linspace(0, 4, 1000)
    analog_signal = (
        2 * np.sin(2 * np.pi * 0.8 * t)
        + 0.5 * np.sin(2 * np.pi * 2.5 * t)
        + 0.3 * np.sin(2 * np.pi * 5 * t)
    )

    # Sampling parameters
    sampling_rate = 8  # samples per second for visualization
    sample_times = np.arange(0, 4, 1 / sampling_rate)
    sampled_values = (
        2 * np.sin(2 * np.pi * 0.8 * sample_times)
        + 0.5 * np.sin(2 * np.pi * 2.5 * sample_times)
        + 0.3 * np.sin(2 * np.pi * 5 * sample_times)
    )

    # Quantization levels (8 levels for clarity)
    quantization_levels = np.linspace(-3, 3, 9)
    quantized_values = np.round(sampled_values * 8 / 6) * 6 / 8

    # Plot analog signal
    ax.plot(t, analog_signal, "gray", linewidth=2, alpha=0.8, label="Analog signal")

    # Plot quantization levels
    for level in quantization_levels:
        ax.axhline(y=level, color="lightblue", linestyle="-", alpha=0.5, linewidth=0.5)

    # Plot sampling times
    for sample_time in sample_times:
        ax.axvline(x=sample_time, color="red", linestyle="--", alpha=0.7, linewidth=1)

    # Plot sampled and quantized points
    ax.scatter(
        sample_times,
        quantized_values,
        color="red",
        s=50,
        zorder=5,
        label="Digital samples",
    )

    # Connect quantized points to show digital signal
    ax.step(
        sample_times,
        quantized_values,
        where="post",
        color="blue",
        linewidth=2,
        alpha=0.8,
        label="Digital signal",
    )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude (normalized)", fontsize=12)
    ax.legend(fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4)
    ax.set_ylim(-3.5, 3.5)

    # Add annotations
    ax.annotate(
        "Quantization levels\n(Bit depth)",
        xy=(0.2, 2.5),
        xytext=(0.5, 2.8),
        arrowprops=dict(arrowstyle="->", color="blue", alpha=0.7),
        fontsize=12,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

    ax.annotate(
        "Sampling rate\n(Time intervals)",
        xy=(1, -2.5),
        xytext=(1.3, -3),
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
        fontsize=12,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sampling_quantization.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated sampling_quantization.pdf")


def create_waveform_spectrograms_figure():
    """
    Create a figure showing waveform, spectrogram, and mel-spectrogram representations.
    """
    # Load an example audio file
    waveform, sr = librosa.load(
        os.path.expanduser(
            "~/msc-thesis/data/datasets/tts_dataset_liepa2_30spk/wavs/L_RA_F4_IS031_02_000189.wav"
        ),
        sr=22050,
    )
    duration = len(waveform) / sr

    # Compute STFT
    stft = librosa.stft(waveform, hop_length=512, n_fft=2048)
    magnitude = np.abs(stft)

    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=80)

    # Create the figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Plot waveform
    time_axis = np.linspace(0, duration, len(waveform))
    axes[0].plot(time_axis, waveform, color="darkblue", linewidth=0.8)
    axes[0].set_title("Raw audio waveform", fontsize=14)
    axes[0].set_ylabel("Amplitude (normalized)", fontsize=12)
    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, duration)
    axes[0].set_ylim(-1, 1)

    # Plot spectrogram
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    times = librosa.frames_to_time(range(magnitude.shape[1]), sr=sr, hop_length=512)

    im1 = axes[1].imshow(
        librosa.amplitude_to_db(magnitude),
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, sr / 2],
        cmap="viridis",
    )
    axes[1].set_title("Linear frequency spectrogram", fontsize=14)
    axes[1].set_ylabel("Frequency (Hz)", fontsize=12)
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylim(0, 8000)  # Focus on relevant frequency range
    axes[1].set_xlim(0, duration)

    # Plot mel-spectrogram
    im2 = axes[2].imshow(
        librosa.power_to_db(mel_spec),
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, 80],
        cmap="viridis",
    )
    axes[2].set_title("Mel-spectrogram", fontsize=14)
    axes[2].set_ylabel("Mel bin index", fontsize=12)
    axes[2].set_xlabel("Time (s)", fontsize=12)
    axes[2].set_xlim(0, duration)

    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=axes[1], format="%d dB")
    cbar1.set_label("Magnitude (dB)", fontsize=11)

    cbar2 = plt.colorbar(im2, ax=axes[2], format="%d dB")
    cbar2.set_label("Power (dB)", fontsize=11)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "waveform_spectrograms.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated waveform_spectrograms.pdf")


def create_speaker_encoder_diagram():
    """
    Create a diagram showing the general architecture of a Speaker Encoder.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Define components and their positions
    components = {
        "input": {
            "pos": (1, 4),
            "size": (1.5, 0.8),
            "label": "Reference Audio\n(Variable Length)",
            "color": "lightblue",
        },
        "preprocessing": {
            "pos": (3.5, 4),
            "size": (1.5, 0.8),
            "label": "Feature\nExtraction",
            "color": "lightgreen",
        },
        "lstm1": {
            "pos": (6, 5),
            "size": (1.2, 0.8),
            "label": "LSTM\nLayer 1",
            "color": "orange",
        },
        "lstm2": {
            "pos": (6, 3.8),
            "size": (1.2, 0.8),
            "label": "LSTM\nLayer 2",
            "color": "orange",
        },
        "lstm3": {
            "pos": (6, 2.6),
            "size": (1.2, 0.8),
            "label": "LSTM\nLayer N",
            "color": "orange",
        },
        "pooling": {
            "pos": (8.5, 4),
            "size": (1.2, 0.8),
            "label": "Global\nPooling",
            "color": "lightyellow",
        },
        "embedding": {
            "pos": (11, 4),
            "size": (1.5, 0.8),
            "label": "Speaker\nEmbedding\n(d-vector)",
            "color": "lightcoral",
        },
    }

    # Draw components
    for comp_name, comp_info in components.items():
        x, y = comp_info["pos"]
        w, h = comp_info["size"]
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=comp_info["color"],
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            comp_info["label"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    # Draw arrows using FancyArrowPatch for precise alignment
    arrows = [
        ((1.75, 4), (2.75, 4)),  # input -> preprocessing
        ((4.25, 4), (5.4, 5)),  # preprocessing -> lstm1
        ((4.25, 4), (5.4, 3.8)),  # preprocessing -> lstm2
        ((4.25, 4), (5.4, 2.6)),  # preprocessing -> lstm3
        ((7.2, 5), (7.9, 4.4)),  # lstm1 -> pooling
        ((7.2, 3.8), (7.9, 4)),  # lstm2 -> pooling
        ((7.2, 2.6), (7.9, 3.6)),  # lstm3 -> pooling
        ((9.1, 4), (10.25, 4)),  # pooling -> embedding
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2,
            color="darkblue",
            zorder=3,
        )
        ax.add_patch(arrow)

    # Add dots to indicate more LSTM layers
    ax.text(6, 2, "⋮", ha="center", va="center", fontsize=20, fontweight="bold")

    # Add dimension annotations
    ax.text(
        1,
        2.8,
        "Variable\nsequence length",
        ha="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.text(
        11,
        2.8,
        "Fixed-length\nvector\n(e.g., 256-dim)",
        ha="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlim(0, 13)
    ax.set_ylim(2, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    # ax.set_title("Speaker Encoder Architecture", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "speaker_encoder_diagram.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("✓ Generated speaker_encoder_diagram.pdf")


def create_tacotron2_architecture():
    """
    Create a diagram showing the Tacotron 2 architecture.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Define main components
    components = {
        # Encoder side
        "text_input": {
            "pos": (2, 1),
            "size": (2, 0.8),
            "label": "Text Input\n(Characters/Phonemes)",
            "color": "lightblue",
        },
        "char_embedding": {
            "pos": (2, 2.5),
            "size": (2, 0.8),
            "label": "Character\nEmbedding",
            "color": "lightgreen",
        },
        "conv_layers": {
            "pos": (2, 4),
            "size": (2, 0.8),
            "label": "Convolutional\nLayers",
            "color": "orange",
        },
        "encoder_lstm": {
            "pos": (2, 5.5),
            "size": (2, 0.8),
            "label": "Bidirectional\nLSTM",
            "color": "yellow",
        },
        # Attention
        "attention": {
            "pos": (6, 4),
            "size": (2, 1.2),
            "label": "Location-Sensitive\nAttention",
            "color": "lightcoral",
        },
        # Decoder side
        "prev_frame": {
            "pos": (10, 1),
            "size": (1.5, 0.8),
            "label": "Previous\nMel Frame",
            "color": "lightgray",
        },
        "prenet": {
            "pos": (10, 2.5),
            "size": (1.5, 0.8),
            "label": "Pre-net\n(FC + Dropout)",
            "color": "lightyellow",
        },
        "decoder_lstm": {
            "pos": (10, 4),
            "size": (1.5, 0.8),
            "label": "Decoder\nLSTM",
            "color": "lightgreen",
        },
        "linear_proj": {
            "pos": (10, 5.5),
            "size": (1.5, 0.8),
            "label": "Linear\nProjection",
            "color": "orange",
        },
        "mel_output": {
            "pos": (10, 7),
            "size": (1.5, 0.8),
            "label": "Mel-spectrogram\nFrame",
            "color": "lightblue",
        },
        # Post-net
        "postnet": {
            "pos": (13, 5.5),
            "size": (1.5, 1.5),
            "label": "Post-net\n(Conv + Residual)",
            "color": "lightsteelblue",
        },
        "final_mel": {
            "pos": (13, 7.5),
            "size": (2, 0.8),
            "label": "Final\nMel-spectrogram",
            "color": "cornflowerblue",
        },
        # Stop token
        "stop_token": {
            "pos": (8, 7),
            "size": (1.2, 0.6),
            "label": "Stop Token\nPrediction",
            "color": "lightpink",
        },
    }

    # Draw components
    for comp_name, comp_info in components.items():
        x, y = comp_info["pos"]
        w, h = comp_info["size"]
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=comp_info["color"],
            edgecolor="black",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            comp_info["label"],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Draw connections using FancyArrowPatch
    arrows = [
        # Encoder path
        ((2, 1.4), (2, 2.1)),  # text -> embedding
        ((2, 2.9), (2, 3.6)),  # embedding -> conv
        ((2, 4.4), (2, 5.1)),  # conv -> lstm
        ((3, 5.5), (5, 4.6)),  # encoder lstm -> attention
        # Decoder path
        ((10, 1.4), (10, 2.1)),  # prev frame -> prenet
        ((10, 2.9), (10, 3.6)),  # prenet -> decoder lstm
        ((10, 4.4), (10, 5.1)),  # decoder lstm -> linear proj
        ((10, 5.9), (10, 6.6)),  # linear proj -> mel output
        # Attention connections
        ((7, 4), (9.25, 4)),  # attention -> decoder lstm
        ((7, 4.6), (9.25, 5.5)),  # attention -> linear proj
        # Post-net
        ((10.75, 7), (12.25, 6.25)),  # mel output -> postnet
        ((13, 6.25), (13, 7.1)),  # postnet -> final mel
        # Stop token
        ((9.25, 5.9), (8.6, 6.7)),  # linear proj -> stop token
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2,
            color="darkblue",
            zorder=3,
        )
        ax.add_patch(arrow)

    # Recurrent connection (curved)
    recurrent_arrow = FancyArrowPatch(
        (9.25, 7),
        (9.25, 1.4),
        arrowstyle="->",
        mutation_scale=20,
        linewidth=2,
        color="red",
        connectionstyle="arc3,rad=-0.3",
        zorder=3,
    )
    ax.add_patch(recurrent_arrow)

    # Add labels for different sections
    ax.text(
        2,
        0.2,
        "ENCODER",
        ha="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )
    ax.text(
        6,
        2.5,
        "ATTENTION",
        ha="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )
    ax.text(
        10,
        0.2,
        "DECODER",
        ha="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    # Add recurrent connection label
    ax.text(
        8.5,
        3.5,
        "Autoregressive\nConnection",
        ha="center",
        fontsize=10,
        style="italic",
        color="red",
        fontweight="bold",
    )

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")
    # ax.set_title(
    #     "Tacotron 2 Architecture with DCA", fontsize=16, fontweight="bold", pad=20
    # )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tacotron2_arch.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated tacotron2_arch.pdf")


def create_glow_tts_architecture():
    """
    Create a diagram showing the Glow-TTS architecture.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Define components
    components = {
        # Input
        "text_input": {
            "pos": (2, 1),
            "size": (2, 0.8),
            "label": "Text Input\n(Phonemes)",
            "color": "lightblue",
        },
        "text_embedding": {
            "pos": (2, 2.5),
            "size": (2, 0.8),
            "label": "Text\nEmbedding",
            "color": "lightgreen",
        },
        # Encoder
        "transformer_enc": {
            "pos": (2, 4.5),
            "size": (2, 1.5),
            "label": "Text Encoder\n(Transformer)",
            "color": "orange",
        },
        # Duration prediction
        "duration_pred": {
            "pos": (5.5, 4.5),
            "size": (1.8, 1.2),
            "label": "Duration\nPredictor\n(Stochastic)",
            "color": "lightyellow",
        },
        # Length regulation
        "length_reg": {
            "pos": (8.5, 4.5),
            "size": (1.5, 0.8),
            "label": "Length\nRegulator",
            "color": "lightsteelblue",
        },
        # Flow decoder
        "flow_block1": {
            "pos": (11.5, 5.5),
            "size": (1.5, 0.8),
            "label": "Flow Block\n(ActNorm + Coupling)",
            "color": "lightcyan",
        },
        "flow_block2": {
            "pos": (11.5, 4.5),
            "size": (1.5, 0.8),
            "label": "Flow Block\n(Affine Coupling)",
            "color": "lightcyan",
        },
        "flow_block3": {
            "pos": (11.5, 3.5),
            "size": (1.5, 0.8),
            "label": "Flow Block\n(1x1 Conv)",
            "color": "lightcyan",
        },
        # Prior and output
        "prior": {
            "pos": (14.5, 3),
            "size": (1.8, 1),
            "label": "Gaussian\nPrior\nz ~ N(0,I)",
            "color": "lavender",
        },
        "mel_output": {
            "pos": (14.5, 5.5),
            "size": (2, 0.8),
            "label": "Mel-spectrogram\nOutput",
            "color": "cornflowerblue",
        },
        # Duration targets (training)
        "duration_target": {
            "pos": (5.5, 1.5),
            "size": (1.5, 0.6),
            "label": "Duration\nTargets\n(Training)",
            "color": "lightgray",
        },
    }

    # Draw components
    for comp_name, comp_info in components.items():
        x, y = comp_info["pos"]
        w, h = comp_info["size"]
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=comp_info["color"],
            edgecolor="black",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            comp_info["label"],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Add dots to indicate more flow blocks
    ax.text(11.5, 2.8, "⋮", ha="center", va="center", fontsize=24, fontweight="bold")

    # Draw connections using FancyArrowPatch
    arrows = [
        # Main forward path
        ((2, 1.4), (2, 2.1)),  # text -> embedding
        ((2, 2.9), (2, 3.75)),  # embedding -> transformer enc
        ((3, 4.5), (4.6, 4.5)),  # transformer enc -> duration pred
        # Duration path
        ((6.4, 4.5), (7.75, 4.5)),  # duration pred -> length reg
        # To decoder
        ((9.25, 4.5), (10.75, 5.5)),  # length reg -> flow block 1
        ((10.75, 4.5), (10.75, 4.5)),  # between flow blocks
        ((12.25, 5.5), (13.5, 5.5)),  # flow blocks -> mel output
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2,
            color="darkblue",
            zorder=3,
        )
        ax.add_patch(arrow)

    # Prior connection (dashed purple)
    prior_arrow = FancyArrowPatch(
        (13.6, 3.5),
        (12.25, 3.9),
        arrowstyle="->",
        mutation_scale=20,
        linewidth=2,
        color="purple",
        linestyle="--",
        zorder=3,
    )
    ax.add_patch(prior_arrow)

    # Training connection (dashed gray)
    training_arrow = FancyArrowPatch(
        (5.5, 1.8),
        (5.5, 3.9),
        arrowstyle="->",
        mutation_scale=20,
        linewidth=2,
        color="gray",
        linestyle="--",
        zorder=3,
    )
    ax.add_patch(training_arrow)

    # Flow connections between blocks (bidirectional)
    flow_arrows = [
        ((11.5, 5.1), (11.5, 4.9)),  # flow1 -> flow2
        ((11.5, 4.1), (11.5, 3.9)),  # flow2 -> flow3
    ]

    for start, end in flow_arrows:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="<->",
            mutation_scale=20,
            linewidth=2,
            color="teal",
            zorder=3,
        )
        ax.add_patch(arrow)

    # Add key features annotations
    ax.text(
        11.5,
        7,
        "Normalizing Flow\n(Invertible Transformations)",
        ha="center",
        fontsize=12,
        style="italic",
        fontweight="bold",
        color="teal",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.7),
    )

    ax.text(
        8.5,
        6.5,
        "Fast Parallel Generation\n(Non-autoregressive)",
        ha="center",
        fontsize=12,
        style="italic",
        fontweight="bold",
        color="green",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    ax.text(
        5.5,
        6.5,
        "Maximum Likelihood\nTraining",
        ha="center",
        fontsize=12,
        style="italic",
        fontweight="bold",
        color="purple",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="plum", alpha=0.7),
    )

    # Add section labels
    ax.text(
        2,
        0.2,
        "ENCODER",
        ha="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )
    ax.text(
        7,
        0.2,
        "DURATION",
        ha="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
    )
    ax.text(
        11.5,
        0.2,
        "FLOW DECODER",
        ha="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.7),
    )
    ax.text(
        14.5,
        0.2,
        "OUTPUT",
        ha="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="cornflowerblue", alpha=0.7),
    )

    ax.set_xlim(0, 17)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    # ax.set_title("Glow-TTS Architecture", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "glow_tts_arch.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated glow_tts_arch.pdf")


def create_latin_square_figure():
    """
    Create a visual representation of Latin square design for subjective evaluation.
    Shows how sentences and models are balanced across listener groups.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Latin square assignment (each model appears once per group, each sentence once per group)
    # Rows = listener groups, columns = presentation order
    latin_square = [
        ["A-S1", "B-S2", "C-S3", "D-S4"],
        ["B-S3", "A-S4", "D-S1", "C-S2"],
        ["C-S4", "D-S3", "A-S2", "B-S1"],
        ["D-S2", "C-S1", "B-S4", "A-S3"],
    ]

    # Color mapping for systems
    colors = {"A": "#FF6B6B", "B": "#4ECDC4", "C": "#45B7D1", "D": "#FFA07A"}

    # Create the grid
    NUM_GROUPS = 4
    for i in range(NUM_GROUPS):
        for j, assignment in enumerate(latin_square[i]):
            system, sentence = assignment.split("-")

            # Draw rectangle with system color
            rect = mpatches.Rectangle(
                (j, 3 - i),
                1,
                1,
                linewidth=2,
                edgecolor="black",
                facecolor=colors[system],
                alpha=0.7,
            )
            ax.add_patch(rect)

            # Add text
            ax.text(
                j + 0.5,
                3 - i + 0.6,
                f"Model {system}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
            ax.text(
                j + 0.5,
                3 - i + 0.3,
                sentence.replace("S", "Sentence "),
                ha="center",
                va="center",
                fontsize=9,
            )

    # Set axis properties
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")

    # Labels
    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels(["Order 1", "Order 2", "Order 3", "Order 4"], fontsize=11)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(["Group 4", "Group 3", "Group 2", "Group 1"], fontsize=11)

    ax.set_xlabel("Presentation order (sequence)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Listener group (ID)", fontsize=12, fontweight="bold")
    # ax.set_title(
    #     "Latin square design for TTS evaluation", fontsize=14, fontweight="bold", pad=20
    # )

    # Add legend
    legend_elements = [
        mpatches.Patch(
            facecolor=colors[s],
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
            label=f"Model {s}",
        )
        for s in ["A", "B", "C", "D"]
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=True,
    )

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "latin_square_design.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated latin_square_design.pdf")


def create_tts_pipeline_figure():
    """
    Create a flowchart showing the TTS pipeline:
    [Text, Speaker Embeddings] -> Acoustic Model -> Mel-Spectrogram -> Vocoder -> Audio
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    # Define box positions and sizes
    boxes = {
        "text": {"pos": (0.3, 5), "width": 1.6, "height": 1, "color": "#E8F4F8"},
        "speaker": {"pos": (0.3, 3.2), "width": 1.6, "height": 1, "color": "#E8F4F8"},
        "acoustic": {"pos": (3, 3.5), "width": 2.5, "height": 2, "color": "#FFE5CC"},
        "mel": {"pos": (6.6, 4.1), "width": 1.6, "height": 0.8, "color": "#E8F4F8"},
        "vocoder": {"pos": (9.2, 3.5), "width": 2.2, "height": 2, "color": "#FFE5CC"},
        "audio": {"pos": (12.3, 4.1), "width": 1.6, "height": 0.8, "color": "#E8F4F8"},
    }

    # Draw boxes
    for name, props in boxes.items():
        x, y = props["pos"]
        w, h = props["width"], props["height"]
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            linewidth=2,
            edgecolor="#2C3E50",
            facecolor=props["color"],
        )
        ax.add_patch(rect)

    # Add text labels
    ax.text(
        1.1, 5.5, "Input text", ha="center", va="center", fontsize=16, fontweight="bold"
    )
    ax.text(
        1.1, 5.1, '"Labas rytas"', ha="center", va="center", fontsize=13, style="italic"
    )

    ax.text(
        1.1,
        3.7,
        "Speaker\nembedding",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(1.1, 3.2, "vector", ha="center", va="center", fontsize=12, style="italic")

    ax.text(
        4.25,
        5.0,
        "Acoustic model",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        4.25,
        4.5,
        "Tacotron 2",
        ha="center",
        va="center",
        fontsize=15,
        style="italic",
        color="#E74C3C",
    )
    ax.text(4.25, 4.2, "or", ha="center", va="center", fontsize=13)
    ax.text(
        4.25,
        3.9,
        "Glow-TTS",
        ha="center",
        va="center",
        fontsize=15,
        style="italic",
        color="#3498DB",
    )

    ax.text(
        7.4,
        4.5,
        "Mel-\nspectrogram",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    ax.text(
        10.3, 5.0, "Vocoder", ha="center", va="center", fontsize=18, fontweight="bold"
    )
    ax.text(
        10.3,
        4.5,
        "Griffin-Lim",
        ha="center",
        va="center",
        fontsize=15,
        style="italic",
        color="#E74C3C",
    )
    ax.text(10.3, 4.2, "or", ha="center", va="center", fontsize=13)
    ax.text(
        10.3,
        3.9,
        "HiFi-GAN",
        ha="center",
        va="center",
        fontsize=15,
        style="italic",
        color="#3498DB",
    )

    ax.text(
        13.1,
        4.5,
        "Audio\nwaveform",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    # Draw arrows using FancyArrowPatch for precise alignment
    arrows = [
        ("text", "acoustic"),
        ("speaker", "acoustic"),
        ("acoustic", "mel"),
        ("mel", "vocoder"),
        ("vocoder", "audio"),
    ]

    for node_from, node_to in arrows:
        start_pos = list(boxes[node_from]["pos"])
        start_pos[0] += boxes[node_from]["width"] + 0.1
        start_pos[1] += boxes[node_from]["height"] / 2

        end_pos = list(boxes[node_to]["pos"])
        end_pos[0] -= 0.05
        end_pos[1] += boxes[node_to]["height"] / 2

        arrow = FancyArrowPatch(
            start_pos,
            end_pos,
            arrowstyle="->",
            mutation_scale=30,
            linewidth=2,
            color="#5E768D",
            zorder=3,
        )
        ax.add_patch(arrow)

    # Set axis limits
    ax.set_xlim(0, 14.2)
    ax.set_ylim(2.5, 6.5)
    ax.set_aspect("equal")

    # # Add title
    # ax.text(
    #     7,
    #     7.0,
    #     "End-to-End Text-to-Speech pipeline",
    #     ha="center",
    #     va="center",
    #     fontsize=21,
    #     fontweight="bold",
    # )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tts_pipeline.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Generated tts_pipeline.pdf")


def main():
    """Generate all thesis figures."""
    print("Generating figures for MSc thesis...")
    print(f"Output directory: {FIGURES_DIR.absolute()}")

    try:
        create_sampling_quantization_figure()
        create_waveform_spectrograms_figure()
        create_speaker_encoder_diagram()
        create_tacotron2_architecture()
        create_glow_tts_architecture()
        create_latin_square_figure()
        create_tts_pipeline_figure()

        print("\n✅ All figures generated successfully!")
        print(f"Generated files:")
        for pdf_file in FIGURES_DIR.glob("*.pdf"):
            print(f"  - {pdf_file.name}")

        print(
            "\nTo include these figures in your LaTeX document, uncomment the \\includegraphics lines in thesis.tex"
        )

    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        raise


if __name__ == "__main__":
    main()
