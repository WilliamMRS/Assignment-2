"""
HOG (Histogram of Oriented Gradients) features with scikit-image.

Steps covered (see assignment HOG tasks 1-4):
  1. compute_hog()            -> HOG descriptor + visualisation for one image
  2. IMAGES                   -> simple -> complex scenes (built-in + vehicle dataset)
  3. plot_pipeline()          -> original | gradient magnitude | HOG image
  4. compare_images()         -> descriptor statistics / similarity between images
     parameter_sweep()        -> effect of cell size, block size, number of bins

Figures are written to image_processing/output/hog/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, data, exposure, filters, io, transform
from skimage.feature import hog

# Paths are relative to this file, so the script runs from any working directory
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "vehicles"
CHOSEN_DIR = HERE / "chosen_images"
OUT_DIR = HERE / "output" / "hog"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All images are resized to the same shape so descriptor lengths are comparable.
# 128 x 192 is divisible by 8 and 16, so every tested cell size tiles exactly.
IMG_SHAPE = (128, 192)

# Default parameters (Dalal & Triggs, 2005)
DEFAULTS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

# (label, loader) ordered from simple to complex.
# horse.png is scikit-image's data.horse(); the others are copied from data/vehicles/.
IMAGES = [
    ("Simple: horse silhouette", lambda: io.imread(CHOSEN_DIR / "horse.png")),
    ("Medium: sedan, clean background", lambda: io.imread(CHOSEN_DIR / "sedan_PIC_1.jpg")),
    ("Complex: motorcycle + rider", lambda: io.imread(CHOSEN_DIR / "motorcycle_PIC_0.jpg")),
    ("Complex: pickup, cluttered scene", lambda: io.imread(CHOSEN_DIR / "pickup_PIC_15.jpg")),
]


# ----------------------------------------------------------------------------
# Step 1: load + preprocess, compute gradients and HOG
# ----------------------------------------------------------------------------
def load_gray(loader):
    """Load an image, convert to grayscale float in [0, 1] and resize."""
    img = loader()
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    img = img.astype(float)
    if img.max() > 1:
        img /= 255.0
    return transform.resize(img, IMG_SHAPE, anti_aliasing=True)


def compute_gradients(img):
    """Per-pixel gradient magnitude and unsigned orientation (0-180 deg).

    Uses the same centred [-1, 0, 1] difference kernel as HOG itself.
    """
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    magnitude = np.hypot(gx, gy)
    orientation = np.rad2deg(np.arctan2(gy, gx)) % 180
    return gx, gy, magnitude, orientation


def compute_hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """Return (feature_vector, hog_visualisation) for a grayscale image."""
    features, hog_image = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    # Stretch the visualisation so weak cells are still visible
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, hog_image.max() or 1))
    return features, hog_image


def expected_length(shape, orientations, pixels_per_cell, cells_per_block):
    """Descriptor length formula: n_blocks_y * n_blocks_x * cells_per_block^2 * bins."""
    cells_y = shape[0] // pixels_per_cell[0]
    cells_x = shape[1] // pixels_per_cell[1]
    blocks_y = cells_y - cells_per_block[0] + 1
    blocks_x = cells_x - cells_per_block[1] + 1
    return blocks_y * blocks_x * cells_per_block[0] * cells_per_block[1] * orientations


# ----------------------------------------------------------------------------
# Steps 2 + 3: apply to several images, visualise original | gradient | HOG
# ----------------------------------------------------------------------------
def plot_pipeline(images):
    fig, axes = plt.subplots(len(images), 4, figsize=(16, 3.4 * len(images)))
    results = {}
    for row, (label, img) in zip(axes, images):
        gx, gy, mag, ori = compute_gradients(img)
        feats, hog_img = compute_hog(img, **DEFAULTS)
        results[label] = dict(img=img, mag=mag, ori=ori, feats=feats)

        row[0].imshow(img, cmap="gray")
        row[0].set_title(label)
        row[1].imshow(mag, cmap="magma")
        row[1].set_title("Gradient magnitude")
        # Orientation shown only where there is a real edge
        ori_masked = np.ma.masked_where(mag < 0.05, ori)
        row[2].imshow(ori_masked, cmap="hsv", vmin=0, vmax=180)
        row[2].set_title("Gradient orientation (edges only)")
        row[3].imshow(hog_img, cmap="gray")
        row[3].set_title(f"HOG image ({feats.size} features)")
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "1_pipeline.png", dpi=120)
    plt.close(fig)
    return results


# ----------------------------------------------------------------------------
# Step 4a: compare HOG descriptors across images
# ----------------------------------------------------------------------------
def orientation_histogram(feats, orientations):
    """Collapse the descriptor into one global histogram over orientation bins."""
    return feats.reshape(-1, orientations).sum(axis=0)


def compare_images(results):
    labels = list(results)
    bins = DEFAULTS["orientations"]
    centres = (np.arange(bins) + 0.5) * 180 / bins

    print("\n=== Descriptor statistics (default params) ===")
    print(f"{'image':36s} {'length':>7s} {'mean':>7s} {'std':>7s} {'% near 0':>9s} {'edge px %':>9s}")
    for label in labels:
        f = results[label]["feats"]
        edge_pct = 100 * (results[label]["mag"] > 0.05).mean()
        print(f"{label:36s} {f.size:7d} {f.mean():7.3f} {f.std():7.3f} "
              f"{100 * (f < 0.01).mean():8.1f}% {edge_pct:8.1f}%")

    # Global orientation histograms
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    width = 180 / bins / (len(labels) + 1)
    for i, label in enumerate(labels):
        h = orientation_histogram(results[label]["feats"], bins)
        axes[0].bar(centres + (i - len(labels) / 2) * width, h / h.sum(), width, label=label)
    axes[0].set_xticks(np.arange(0, 181, 20))
    axes[0].set_xlabel("Gradient orientation (deg, 0 = vertical edge, 90 = horizontal edge)")
    axes[0].set_ylabel("Share of HOG energy")
    axes[0].set_title("Global orientation distribution")
    axes[0].legend(fontsize=8)

    # Cosine similarity between descriptors
    F = np.stack([results[l]["feats"] for l in labels])
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    sim = F @ F.T
    im = axes[1].imshow(sim, cmap="viridis", vmin=0, vmax=1)
    short = [l.split(":")[1].strip() for l in labels]
    axes[1].set_xticks(range(len(labels)), short, rotation=30, ha="right", fontsize=8)
    axes[1].set_yticks(range(len(labels)), short, fontsize=8)
    for (r, c), v in np.ndenumerate(sim):
        axes[1].text(c, r, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.6 else "k")
    axes[1].set_title("Cosine similarity of HOG descriptors")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "2_comparison.png", dpi=120)
    plt.close(fig)

    print("\nCosine similarity matrix:")
    print(np.array2string(sim, precision=2))


def class_similarity(n_per_class=30, seed=0):
    """Does HOG separate vehicle classes? Mean cosine similarity within/between classes."""
    rng = np.random.default_rng(seed)
    classes = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    feats = {}
    for c in classes:
        files = sorted((DATA_DIR / c).glob("*.[jJ][pP][gG]"))
        pick = rng.choice(len(files), size=min(n_per_class, len(files)), replace=False)
        F = np.stack([compute_hog(load_gray(lambda f=files[i]: io.imread(f)), **DEFAULTS)[0]
                      for i in pick])
        feats[c] = F / np.linalg.norm(F, axis=1, keepdims=True)

    M = np.zeros((len(classes), len(classes)))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            s = feats[a] @ feats[b].T
            if i == j:  # exclude self-similarity on the diagonal
                s = s[~np.eye(len(s), dtype=bool)]
            M[i, j] = s.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(len(classes)), classes, rotation=30)
    ax.set_yticks(range(len(classes)), classes)
    for (r, c), v in np.ndenumerate(M):
        ax.text(c, r, f"{v:.3f}", ha="center", va="center", color="w")
    ax.set_title(f"Mean HOG cosine similarity\n({n_per_class} random images per class)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3_class_similarity.png", dpi=120)
    plt.close(fig)

    print("\n=== Mean within/between-class similarity ===")
    print("        " + " ".join(f"{c[:8]:>9s}" for c in classes))
    for c, row in zip(classes, M):
        print(f"{c[:8]:8s}" + " ".join(f"{v:9.3f}" for v in row))


# ----------------------------------------------------------------------------
# Step 4b: parameter sweep - cell size, block size, number of bins
# ----------------------------------------------------------------------------
SWEEPS = {
    "cell size (pixels_per_cell)": ("pixels_per_cell", [(4, 4), (8, 8), (16, 16)]),
    "block size (cells_per_block)": ("cells_per_block", [(1, 1), (2, 2), (4, 4)]),
    "number of bins (orientations)": ("orientations", [3, 9, 18]),
}


def parameter_sweep(img, label):
    print(f"\n=== Parameter sweep on '{label}' ===")
    fig, axes = plt.subplots(len(SWEEPS), 3, figsize=(13, 3.2 * len(SWEEPS)))
    for row, (title, (key, values)) in zip(axes, SWEEPS.items()):
        for ax, v in zip(row, values):
            params = {**DEFAULTS, key: v}
            feats, hog_img = compute_hog(img, **params)
            assert feats.size == expected_length(img.shape, **params)
            ax.imshow(hog_img, cmap="gray")
            ax.set_title(f"{key}={v}\nlength={feats.size}", fontsize=10)
            ax.axis("off")
            print(f"{key:16s} = {str(v):9s} -> length {feats.size:6d}")
        row[0].text(-0.08, 0.5, title, transform=row[0].transAxes, rotation=90,
                    ha="right", va="center", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "4_parameter_sweep.png", dpi=120)
    plt.close(fig)


def bins_histogram_detail(img):
    """Show one cell's histogram with different bin counts: resolution vs. sparsity."""
    _, _, mag, ori = compute_gradients(img)
    # Pick the 16x16 patch with the most edge energy
    h, w = img.shape
    best = max(((y, x) for y in range(0, h - 16, 8) for x in range(0, w - 16, 8)),
               key=lambda p: mag[p[0]:p[0] + 16, p[1]:p[1] + 16].sum())
    y, x = best
    m, o = mag[y:y + 16, x:x + 16].ravel(), ori[y:y + 16, x:x + 16].ravel()

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    axes[0].imshow(img, cmap="gray")
    axes[0].add_patch(plt.Rectangle((x, y), 16, 16, ec="red", fc="none", lw=2))
    axes[0].set_title("Patch used (red)")
    axes[0].axis("off")
    for ax, b in zip(axes[1:], [3, 9, 18]):
        hist, edges = np.histogram(o, bins=b, range=(0, 180), weights=m)
        ax.bar(edges[:-1], hist, width=180 / b, align="edge", edgecolor="k")
        ax.set_title(f"{b} bins ({180 / b:.0f} deg each)")
        ax.set_xlabel("orientation (deg)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "5_bins_detail.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    images = [(label, load_gray(loader)) for label, loader in IMAGES]

    results = plot_pipeline(images)
    compare_images(results)
    class_similarity()

    label, img = images[1]  # sedan: clear structure, good for showing parameter effects
    parameter_sweep(img, label)
    bins_histogram_detail(img)

    print(f"\nFigures saved to {OUT_DIR.resolve()}")
