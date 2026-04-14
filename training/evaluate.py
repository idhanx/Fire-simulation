"""
Forest Fire — Evaluation Pipeline (Final)

Threshold strategy: best F1 where precision > 0.8 AND recall > 0.8.
Outputs: PR curve, confusion matrix, metrics table, best threshold file.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.cnn import FireModel

OUT_DIR = "outputs/evaluation"


# ── Metrics ──────────────────────────────────────────────────
def compute_metrics(preds, labels, threshold):
    """Binary classification metrics at a given threshold."""
    p = (preds >= threshold).astype(int)
    tp = int(((p == 1) & (labels == 1)).sum())
    tn = int(((p == 0) & (labels == 0)).sum())
    fp = int(((p == 1) & (labels == 0)).sum())
    fn = int(((p == 0) & (labels == 1)).sum())
    total = tp + tn + fp + fn + 1e-8
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return dict(t=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
                acc=(tp+tn)/total, prec=prec, rec=rec, f1=f1)


def find_best_threshold(preds, labels):
    """
    Sweep thresholds 0.05–0.95.
    Pick best F1 among those with precision > 0.8 AND recall > 0.8.
    Fallback to overall best F1 if no candidate meets the constraint.
    """
    thresholds = np.arange(0.05, 0.96, 0.025)
    results = [compute_metrics(preds, labels, t) for t in thresholds]

    # Constrained: P > 0.8 and R > 0.8
    valid = [r for r in results if r["prec"] > 0.8 and r["rec"] > 0.8]
    if valid:
        best = max(valid, key=lambda r: r["f1"])
    else:
        best = max(results, key=lambda r: r["f1"])
        print("  (!) No threshold with P>0.8 & R>0.8 — using unconstrained best F1")

    return best, results


# ── Plots ────────────────────────────────────────────────────
def save_pr_curve(results, path):
    prec = [r["prec"] for r in results]
    rec  = [r["rec"]  for r in results]
    ts   = [r["t"]    for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    for f1v in [0.2, 0.4, 0.6, 0.8]:
        x = np.linspace(0.01, 1, 100)
        y = f1v * x / (2 * x - f1v + 1e-8)
        m = (y > 0) & (y <= 1)
        ax.plot(x[m], y[m], "--", color="gray", alpha=0.3)
    ax.plot(rec, prec, "b-o", ms=3, lw=2, label="PR Curve")
    for i, t in enumerate(ts):
        if i % 3 == 0:
            ax.annotate(f"{t:.2f}", (rec[i], prec[i]), fontsize=6)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve",
           xlim=(0, 1.05), ylim=(0, 1.05))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def save_confusion_matrix(m, path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}\n({100*cm[i,j]/cm.sum():.1f}%)",
                    ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=["No Fire","Fire"], yticklabels=["No Fire","Fire"],
           xlabel="Predicted", ylabel="Actual",
           title=f"Confusion Matrix (t={m['t']:.2f})")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


def save_comparison(labels_2d, preds, best, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(labels_2d, cmap="Reds"); axes[0].set_title("Ground Truth")
    axes[1].imshow(preds, cmap="hot", vmin=0, vmax=1); axes[1].set_title("Probabilities")
    axes[2].imshow((preds >= best["t"]).astype(int), cmap="Reds")
    axes[2].set_title(f"Thresholded (t={best['t']:.2f})")
    for ax in axes: ax.axis("off")
    fig.suptitle("Fire Prediction Evaluation", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────
def evaluate():
    print("=" * 55)
    print("  FOREST FIRE — EVALUATION")
    print("=" * 55)

    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FireModel()
    model.load_state_dict(torch.load("outputs/model.pth", map_location=device,
                                     weights_only=True))
    model.to(device).eval()

    # Load data + predict
    features = np.load("data/processed/features.npy")
    labels   = np.load("data/labels/labels.npy").squeeze()
    feat_t   = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = torch.sigmoid(model(feat_t)).squeeze().cpu().numpy()

    print(f"\n  Pred range: [{preds.min():.4f}, {preds.max():.4f}]  mean={preds.mean():.4f}")

    pf, lf = preds.flatten(), labels.flatten()

    # Threshold search (constrained)
    best, all_results = find_best_threshold(pf, lf)

    # Metrics table
    print(f"\n{'Thresh':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Acc':>7}")
    print("-" * 40)
    for r in all_results:
        tag = " *" if r["t"] == best["t"] else ""
        print(f"  {r['t']:.3f}  {r['prec']:.4f}  {r['rec']:.4f}  {r['f1']:.4f}  {r['acc']:.4f}{tag}")

    print(f"\n  BEST t={best['t']:.3f}  P={best['prec']:.4f}  R={best['rec']:.4f}  "
          f"F1={best['f1']:.4f}  Acc={best['acc']:.4f}")
    print(f"  TP={best['tp']:,}  FP={best['fp']:,}  FN={best['fn']:,}  TN={best['tn']:,}")

    # Save outputs
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(f"{OUT_DIR}/best_threshold.txt", "w") as f:
        f.write(str(best["t"]))
    print(f"\n  Threshold saved → {OUT_DIR}/best_threshold.txt")

    save_pr_curve(all_results, f"{OUT_DIR}/pr_curve.png")
    save_confusion_matrix(best, f"{OUT_DIR}/confusion_matrix.png")
    save_comparison(labels, preds, best, f"{OUT_DIR}/prediction_comparison.png")
    np.save(f"{OUT_DIR}/prediction_probs.npy", preds)

    print("\n  Done.")


if __name__ == "__main__":
    evaluate()