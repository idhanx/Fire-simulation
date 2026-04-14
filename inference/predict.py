"""
Forest Fire — Inference Pipeline (Final)

Loads optimal threshold from evaluation.
Saves: probability map, binary mask, confidence map.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.cnn import FireModel

OUT_DIR = "outputs/predictions"


def load_threshold(path="outputs/evaluation/best_threshold.txt", default=0.5):
    if os.path.exists(path):
        t = float(open(path).read().strip())
        print(f"  Threshold: {t:.3f} (from {path})")
        return t
    print(f"  Threshold: {default} (default — run evaluate.py first)")
    return default


def predict():
    print("=" * 55)
    print("  FOREST FIRE — INFERENCE")
    print("=" * 55)

    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FireModel()
    model.load_state_dict(torch.load("outputs/model.pth", map_location=device,
                                     weights_only=True))
    model.to(device).eval()

    # Load data + predict
    features = np.load("data/processed/features.npy")
    feat_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(feat_t)).squeeze().cpu().numpy()

    # Threshold + binary mask
    threshold = load_threshold()
    mask = (probs >= threshold).astype(np.uint8)

    # Confidence map: how far from decision boundary (0.5)
    confidence = np.abs(probs - 0.5)

    # Stats
    fire_px = mask.sum()
    total_px = mask.size
    print(f"\n  Fire pixels : {fire_px:,} / {total_px:,} ({100*fire_px/total_px:.2f}%)")
    print(f"  Prob range  : [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  Confidence  : [{confidence.min():.4f}, {confidence.max():.4f}]")

    # Save arrays
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(f"{OUT_DIR}/fire_map.npy", probs)
    np.save(f"{OUT_DIR}/fire_mask.npy", mask)
    np.save(f"{OUT_DIR}/confidence.npy", confidence)

    # Visualization (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    im0 = axes[0].imshow(probs, cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("Probability Map")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    axes[1].imshow(mask, cmap="Reds")
    axes[1].set_title(f"Binary Mask (t={threshold:.2f})")

    im2 = axes[2].imshow(confidence, cmap="viridis", vmin=0, vmax=0.5)
    axes[2].set_title("Confidence Map")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes: ax.axis("off")
    fig.suptitle("Forest Fire Prediction", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fire_prediction.png", dpi=150)
    plt.close(fig)

    print(f"\n  Saved → {OUT_DIR}/")
    print(f"    fire_map.npy       (probabilities)")
    print(f"    fire_mask.npy      (binary mask)")
    print(f"    confidence.npy     (confidence map)")
    print(f"    fire_prediction.png (visualization)")
    print("\n  Done.")


if __name__ == "__main__":
    predict()